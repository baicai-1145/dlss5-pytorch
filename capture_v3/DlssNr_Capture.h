// DlssNr_Capture.h — Capture v3
//
// v3 adds what a weight-level reimplementation actually needs: the MODEL'S RAW OUTPUT
// (g_nr.output, before the resolve composition) and the resolve's untouched input
// (hdrCopy = the frame exactly as the game produced it). v2's "after" is the resolve
// composition — lerp(original, HueOkLab(model*ratio), Transfer/ColourStrength) — which
// is NOT the network's answer. Fitting weights against it conflates the model with
// RenoDX's post shader.
//
// Planes per frame in v3:
//   before      = colorCopy   (the proxy the model was shown; SDR => passthrough copy)
//   model_raw   = g_nr.output (the network's answer, pre-resolve)          [NEW]
//   hdr_copy    = hdrCopy     (untouched game frame, linear if HDR)        [NEW]
//   after       = target      (final frame after the resolve composition)
//   model_input = same buffer as "before" (kept for v2 compat)
//   depth / motion = guides
//
// Manifest gains: passthrough flag, whitePoint, transferStrength, colourStrength,
// intensity, style, localStructure/Tone/Skin, autoMask, per-frame reset flag, and the
// resolve parameter block. Everything a bit-exact resolve reimplementation needs.

#pragma once

#include <windows.h>
#include <d3d12.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace capture
{
// 16 frames: enough to observe the recurrent path settle after a cold start
// (reset=1 frame followed by accumulation frames).
constexpr unsigned int kMaxFrames = 16;

struct Shot
{
    ID3D12Resource* readback = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout = {};
    unsigned long long bytes = 0;
};

struct FrameFlags
{
    unsigned long long frameIndex = 0;
    int reset = 0;             // 1 if this evaluate ran with DLSSNR.Reset=1 (cold start)
};

class FrameCapture
{
  public:
    void request(unsigned int frames)
    {
        if (active_)
            return;

        wanted_ = frames > kMaxFrames ? kMaxFrames : frames;
        captured_ = 0;
        active_ = wanted_ > 0;
    }

    bool isActive() const { return active_; }
    unsigned int progress() const { return captured_; }

    // v3: record(before, modelRaw, hdrCopy, after, ...). modelRaw/hdrCopy may be null
    // on paths that never ran the model; those frames simply omit the new planes.
    void record(ID3D12GraphicsCommandList* cmd, ID3D12Device* device, ID3D12Resource* before,
                D3D12_RESOURCE_STATES beforeState, ID3D12Resource* modelRaw,
                ID3D12Resource* hdrCopy, ID3D12Resource* after, D3D12_RESOURCE_STATES afterState,
                unsigned long long frameIndex, int reset)
    {
        if (!active_ || before == nullptr || after == nullptr)
            return;

        if (ready_ || captured_ >= wanted_)
            return;

        if (!ensure(device, before, modelRaw, hdrCopy, after))
        {
            active_ = false;
            return;
        }

        if (captured_ >= beforeShots_.size() || captured_ >= afterShots_.size())
        {
            ready_ = true;
            return;
        }

        copy(cmd, before, beforeState, beforeShots_[captured_]);

        if (modelRaw != nullptr && captured_ < modelShots_.size() &&
            modelShots_[captured_].readback != nullptr)
        {
            // After evaluate the model output sits in UNORDERED_ACCESS; the caller
            // transitions it to SRV for the resolve. Record happens in that SRV window:
            // the caller must call record() after its UAV->SRV barrier on the output.
            copy(cmd, modelRaw, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                 modelShots_[captured_]);
        }

        if (hdrCopy != nullptr && captured_ < hdrShots_.size() &&
            hdrShots_[captured_].readback != nullptr)
        {
            copy(cmd, hdrCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                 hdrShots_[captured_]);
        }

        copy(cmd, after, afterState, afterShots_[captured_]);

        if (captured_ < guideShots_[0].size())
        {
            for (size_t g = 0; g < guideShots_.size(); ++g)
                if (guideShots_[g][captured_].readback != nullptr)
                    copy(cmd, guideResources_[g], guideStates_[g], guideShots_[g][captured_]);
        }

        flags_[captured_] = { frameIndex, reset };
        ++captured_;

        if (captured_ >= wanted_)
            ready_ = true;
    }

    void setGuides(ID3D12Device* device, const std::vector<ID3D12Resource*>& resources)
    {
        if (!guideShots_.empty())
            return;

        guideResources_ = resources;
        guideStates_.assign(resources.size(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        guideShots_.resize(resources.size());

        for (size_t g = 0; g < resources.size(); ++g)
        {
            if (resources[g] == nullptr)
                continue;

            guideShots_[g].resize(wanted_);

            D3D12_RESOURCE_DESC desc = resources[g]->GetDesc();
            desc.Format = TypedForCopy(desc.Format);
            guideDescs_.push_back(desc);

            for (unsigned int i = 0; i < wanted_; ++i)
                alloc(device, desc, guideShots_[g][i]);
        }
    }

    bool readyToWrite() const { return ready_; }

    std::string write(const std::filesystem::path& directory)
    {
        if (!ready_)
            return {};

        if (captured_ > 0 && isDark(beforeShots_[0]))
        {
            const unsigned int frames = wanted_;
            release();
            request(frames);
            return {};
        }

        std::error_code ec;
        std::filesystem::create_directories(directory, ec);

        for (unsigned int i = 0; i < captured_; ++i)
        {
            dump(directory, "before", i, beforeShots_[i]);
            dump(directory, "model_raw", i, modelShots_[i]);
            dump(directory, "hdr_copy", i, hdrShots_[i]);
            dump(directory, "after", i, afterShots_[i]);

            static const char* guideNames[] = { "model_input", "depth", "motion" };
            for (size_t g = 0; g < guideShots_.size() && g < 3; ++g)
                if (i < guideShots_[g].size() && guideShots_[g][i].readback != nullptr)
                    dump(directory, guideNames[g], i, guideShots_[g][i]);
        }

        writeManifest(directory);
        release();
        return directory.string();
    }

    void release()
    {
        for (auto& s : beforeShots_)
            if (s.readback != nullptr)
                s.readback->Release();

        for (auto& s : modelShots_)
            if (s.readback != nullptr)
                s.readback->Release();

        for (auto& s : hdrShots_)
            if (s.readback != nullptr)
                s.readback->Release();

        for (auto& s : afterShots_)
            if (s.readback != nullptr)
                s.readback->Release();

        for (auto& shots : guideShots_)
            for (auto& s : shots)
                if (s.readback != nullptr)
                    s.readback->Release();

        beforeShots_.clear();
        modelShots_.clear();
        hdrShots_.clear();
        afterShots_.clear();
        guideShots_.clear();
        guideResources_.clear();
        guideStates_.clear();
        guideDescs_.clear();
        scalars_.clear();
        active_ = false;
        ready_ = false;
        captured_ = 0;
    }

    void setScalar(const std::string& name, double value) { scalars_.push_back({ name, value }); }

  private:
    bool ensure(ID3D12Device* device, ID3D12Resource* before, ID3D12Resource* modelRaw,
                ID3D12Resource* hdrCopy, ID3D12Resource* after)
    {
        if (!beforeShots_.empty())
            return true;

        beforeShots_.resize(wanted_);
        modelShots_.resize(wanted_);
        hdrShots_.resize(wanted_);
        afterShots_.resize(wanted_);
        flags_.resize(wanted_);

        beforeDesc_ = before->GetDesc();
        afterDesc_ = after->GetDesc();
        beforeDesc_.Format = TypedForCopy(beforeDesc_.Format);
        afterDesc_.Format = TypedForCopy(afterDesc_.Format);

        // The model output and hdr copy share the work-resolution description of the
        // output when present; fall back to the before-desc (SDR passthrough case, where
        // every surface is the same size and format).
        D3D12_RESOURCE_DESC modelDesc = beforeDesc_;
        if (modelRaw != nullptr)
        {
            modelDesc = modelRaw->GetDesc();
            modelDesc.Format = TypedForCopy(modelDesc.Format);
        }
        modelDesc_ = modelDesc;

        D3D12_RESOURCE_DESC hdrDesc = beforeDesc_;
        if (hdrCopy != nullptr)
        {
            hdrDesc = hdrCopy->GetDesc();
            hdrDesc.Format = TypedForCopy(hdrDesc.Format);
        }
        hdrDesc_ = hdrDesc;

        for (unsigned int i = 0; i < wanted_; ++i)
        {
            if (!alloc(device, beforeDesc_, beforeShots_[i]) ||
                !alloc(device, afterDesc_, afterShots_[i]))
                return false;

            // New planes are best-effort: a null model output (model refused) or a
            // mismatched hdr copy must not kill the whole capture.
            alloc(device, modelDesc_, modelShots_[i]);
            alloc(device, hdrDesc_, hdrShots_[i]);
        }

        return true;
    }

    static DXGI_FORMAT TypedForCopy(DXGI_FORMAT f)
    {
        switch (f)
        {
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
            return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case DXGI_FORMAT_R32G32B32A32_TYPELESS:
            return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
            return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS:
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_R11G11B10_TYPELESS:
            return DXGI_FORMAT_R11G11B10_FLOAT;
        default:
            return f;
        }
    }

    static bool alloc(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, Shot& shot)
    {
        D3D12_RESOURCE_DESC typed = desc;
        typed.Format = TypedForCopy(desc.Format);

        unsigned long long total = 0;
        device->GetCopyableFootprints(&typed, 0, 1, 0, &shot.layout, nullptr, nullptr, &total);
        shot.bytes = total;

        if (total == 0)
            return false;

        D3D12_HEAP_PROPERTIES heap = {};
        heap.Type = D3D12_HEAP_TYPE_READBACK;

        D3D12_RESOURCE_DESC buf = {};
        buf.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buf.Width = total;
        buf.Height = 1;
        buf.DepthOrArraySize = 1;
        buf.MipLevels = 1;
        buf.Format = DXGI_FORMAT_UNKNOWN;
        buf.SampleDesc.Count = 1;
        buf.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        return SUCCEEDED(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &buf, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&shot.readback)));
    }

    static void copy(ID3D12GraphicsCommandList* cmd, ID3D12Resource* src, D3D12_RESOURCE_STATES state, Shot& shot)
    {
        if (shot.readback == nullptr)
            return;

        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = src;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = state;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;

        const bool needsTransition = state != D3D12_RESOURCE_STATE_COPY_SOURCE;

        if (needsTransition)
            cmd->ResourceBarrier(1, &b);

        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = shot.readback;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = shot.layout;

        D3D12_TEXTURE_COPY_LOCATION source = {};
        source.pResource = src;
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        source.SubresourceIndex = 0;

        cmd->CopyTextureRegion(&dst, 0, 0, 0, &source, nullptr);

        if (needsTransition)
        {
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            b.Transition.StateAfter = state;
            cmd->ResourceBarrier(1, &b);
        }
    }

    static bool isDark(Shot& shot)
    {
        if (shot.readback == nullptr || shot.bytes == 0)
            return false;

        void* mapped = nullptr;
        D3D12_RANGE range = { 0, (SIZE_T) shot.bytes };

        if (FAILED(shot.readback->Map(0, &range, &mapped)) || mapped == nullptr)
            return false;

        const unsigned char* p = (const unsigned char*) mapped;
        unsigned long long total = 0;
        unsigned long long count = 0;

        for (unsigned long long i = 0; i < shot.bytes; i += 1021)
        {
            total += p[i];
            ++count;
        }

        D3D12_RANGE written = { 0, 0 };
        shot.readback->Unmap(0, &written);
        return count > 0 && total / count < 4;
    }

    static void dump(const std::filesystem::path& dir, const char* which, unsigned int index, Shot& shot)
    {
        if (shot.readback == nullptr)
            return;

        void* mapped = nullptr;
        D3D12_RANGE range = { 0, (SIZE_T) shot.bytes };

        if (FAILED(shot.readback->Map(0, &range, &mapped)) || mapped == nullptr)
            return;

        char name[64];
        std::snprintf(name, sizeof(name), "%s_%02u.raw", which, index);
        const auto path = dir / name;

        if (std::FILE* f = _wfopen(path.wstring().c_str(), L"wb"))
        {
            std::fwrite(mapped, 1, (size_t) shot.bytes, f);
            std::fclose(f);
        }

        D3D12_RANGE written = { 0, 0 };
        shot.readback->Unmap(0, &written);
    }

    void writeManifest(const std::filesystem::path& dir)
    {
        const auto path = dir / "manifest.txt";

        if (std::FILE* f = _wfopen(path.wstring().c_str(), L"wt"))
        {
            std::fprintf(f, "version 3\n");
            std::fprintf(f, "frames %u\n", captured_);
            std::fprintf(f, "before width %llu height %u format %d rowPitch %u\n",
                         (unsigned long long) beforeDesc_.Width, beforeDesc_.Height, (int) beforeDesc_.Format,
                         beforeShots_.empty() ? 0 : beforeShots_[0].layout.Footprint.RowPitch);
            std::fprintf(f, "model_raw width %llu height %u format %d rowPitch %u\n",
                         (unsigned long long) modelDesc_.Width, modelDesc_.Height, (int) modelDesc_.Format,
                         modelShots_.empty() ? 0 : modelShots_[0].layout.Footprint.RowPitch);
            std::fprintf(f, "hdr_copy width %llu height %u format %d rowPitch %u\n",
                         (unsigned long long) hdrDesc_.Width, hdrDesc_.Height, (int) hdrDesc_.Format,
                         hdrShots_.empty() ? 0 : hdrShots_[0].layout.Footprint.RowPitch);
            std::fprintf(f, "after width %llu height %u format %d rowPitch %u\n",
                         (unsigned long long) afterDesc_.Width, afterDesc_.Height, (int) afterDesc_.Format,
                         afterShots_.empty() ? 0 : afterShots_[0].layout.Footprint.RowPitch);
            std::fprintf(f, "\nmodel_raw_NN.raw is the network's own answer, before the resolve.\n");
            std::fprintf(f, "hdr_copy_NN.raw is the untouched game frame the encode read.\n");
            std::fprintf(f, "after_NN.raw is the final frame after the resolve composition.\n");

            if (!guideDescs_.empty())
            {
                static const char* guideNames[] = { "model_input", "depth", "motion" };
                for (size_t g = 0; g < guideDescs_.size() && g < 3; ++g)
                    std::fprintf(f, "%s width %llu height %u format %d rowPitch %u\n", guideNames[g],
                                 (unsigned long long) guideDescs_[g].Width, guideDescs_[g].Height,
                                 (int) guideDescs_[g].Format,
                                 guideShots_[g].empty() ? 0 : guideShots_[g][0].layout.Footprint.RowPitch);
            }

            for (unsigned int i = 0; i < captured_ && i < flags_.size(); ++i)
                std::fprintf(f, "frame %u index %llu reset %d\n", i, flags_[i].frameIndex,
                             flags_[i].reset);

            for (const auto& s : scalars_)
                std::fprintf(f, "param %s %g\n", s.first.c_str(), s.second);

            std::fclose(f);
        }
    }

    std::vector<Shot> beforeShots_;
    std::vector<Shot> modelShots_;
    std::vector<Shot> hdrShots_;
    std::vector<Shot> afterShots_;
    std::vector<FrameFlags> flags_;
    std::vector<std::vector<Shot>> guideShots_;
    std::vector<ID3D12Resource*> guideResources_;
    std::vector<D3D12_RESOURCE_STATES> guideStates_;
    std::vector<D3D12_RESOURCE_DESC> guideDescs_;
    std::vector<std::pair<std::string, double>> scalars_;
    D3D12_RESOURCE_DESC beforeDesc_ = {};
    D3D12_RESOURCE_DESC modelDesc_ = {};
    D3D12_RESOURCE_DESC hdrDesc_ = {};
    D3D12_RESOURCE_DESC afterDesc_ = {};
    unsigned int wanted_ = 0;
    unsigned int captured_ = 0;
    bool active_ = false;
    bool ready_ = false;
};
} // namespace capture
