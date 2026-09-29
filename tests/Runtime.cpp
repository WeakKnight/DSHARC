// D3D12 compute tests for the real HLSL implementation. No ray tracing scene.
// Build from an x64 VS developer shell; see Runtime.ps1.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
using U = uint32_t;
struct F4 { float x, y, z, w; };
struct U4 { U x, y, z, w; };
struct Surface { float x, y, z, pad0, nx, ny, nz, pad1; };
struct Material { float r, g, b; U valid; float er, eg, eb; U pad; };
struct Constants {
    float cx = 0, cy = 0, cz = 0, cell = 1;
    float ox = 0, oy = 0, oz = 0, distance = 16;
    U level = 0, capacity = 64, frame = 0, stale = 16;
    U accumulation = 32, groupsX = 1, requests = 1024, pad = 0;
};
static_assert(sizeof(Surface) == 32 && sizeof(Material) == 32 && sizeof(Constants) == 64);

void Check(HRESULT hr) {
    if (FAILED(hr)) throw std::runtime_error("D3D12 HRESULT " + std::to_string(U(hr)));
}
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
bool Near(float a, float b) { return std::abs(a - b) < 0.0001f; }
Surface Hit(float x, float nz = 1) { return {x, .25f, .25f, 0, 0, 0, nz, 0}; }

class TestGPU {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12RootSignature> root;
    std::map<std::string, ComPtr<ID3D12PipelineState>> pipelines;
    U descriptorSize = 0;
    uint64_t fenceValue = 0;
    HANDLE eventHandle = nullptr;
    std::filesystem::path directory;
    std::array<ComPtr<ID3D12Resource>, 16> buffers;
    std::array<U, 16> strides = {8,16,32,32,16,16,16,4,4,4,4,16,32,32,16,16};
    std::array<U, 16> counts = {64,64,64,64,64,64,64,64,1,3,1024,1024,1024,64,64,64};

    ComPtr<ID3D12Resource> Buffer(uint64_t bytes, D3D12_HEAP_TYPE type) {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = type;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = bytes; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
        d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        d.Flags = type == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        auto state = type == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS :
            type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST;
        ComPtr<ID3D12Resource> resource;
        Check(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&resource)));
        return resource;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE CPU(U index) {
        auto handle = heap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += SIZE_T(index) * descriptorSize; return handle;
    }
    void UAV(U index) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
        view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        view.Buffer.NumElements = counts[index];
        view.Buffer.StructureByteStride = strides[index];
        if (index == 9) {
            view.Format = DXGI_FORMAT_R32_TYPELESS;
            view.Buffer.StructureByteStride = 0;
            view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        }
        device->CreateUnorderedAccessView(buffers[index].Get(), nullptr, &view, CPU(index));
    }
    void Open() { Check(allocator->Reset()); Check(list->Reset(allocator.Get(), nullptr)); }
    void Submit() {
        Check(list->Close()); ID3D12CommandList* commands[] = {list.Get()};
        queue->ExecuteCommandLists(1, commands);
        Check(queue->Signal(fence.Get(), ++fenceValue));
        Check(fence->SetEventOnCompletion(fenceValue, eventHandle));
        Require(WaitForSingleObject(eventHandle, 30000) == WAIT_OBJECT_0, "GPU test timed out");
        Check(device->GetDeviceRemovedReason());
    }
public:
    Constants constants;
    explicit TestGPU(const std::filesystem::path& path) : directory(path) {
        // Enable API validation when Graphics Tools is installed.
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
        Check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
        D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
        Check(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model)));
        Require(model.HighestShaderModel >= D3D_SHADER_MODEL_6_6, "SM 6.6 device required");
        D3D12_COMMAND_QUEUE_DESC q{}; q.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        Check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)));
        Check(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&allocator)));
        Check(device->CreateCommandList(0, q.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
        Check(list->Close()); Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        Require(eventHandle != nullptr, "CreateEvent failed");
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 16;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        Check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
        descriptorSize = device->GetDescriptorHandleIncrementSize(hd.Type);
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[0].NumDescriptors = 12;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[1].NumDescriptors = 4;
        D3D12_ROOT_PARAMETER parameters[3]{};
        for (U i = 0; i < 2; ++i) {
            parameters[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[i].DescriptorTable = {1, &ranges[i]};
        }
        parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[2].Constants = {0, 0, 16};
        D3D12_ROOT_SIGNATURE_DESC rs{}; rs.NumParameters = 3; rs.pParameters = parameters;
        ComPtr<ID3DBlob> blob, error;
        Check(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
        Check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)));
        for (U i = 0; i < 16; ++i) {
            buffers[i] = Buffer(uint64_t(strides[i]) * counts[i], i < 12 ? D3D12_HEAP_TYPE_DEFAULT : D3D12_HEAP_TYPE_UPLOAD);
            if (i < 12) UAV(i);
            else {
                D3D12_SHADER_RESOURCE_VIEW_DESC view{};
                view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                view.Buffer.NumElements = counts[i]; view.Buffer.StructureByteStride = strides[i];
                device->CreateShaderResourceView(buffers[i].Get(), &view, CPU(i));
            }
        }
    }
    ~TestGPU() { if (eventHandle) CloseHandle(eventHandle); }
    template<class T> void Upload(U slot, const std::vector<T>& values) {
        Require(slot >= 12 && sizeof(T) * values.size() <= uint64_t(strides[slot]) * counts[slot], "Bad upload size");
        void* data; D3D12_RANGE range{0, 0};
        Check(buffers[slot]->Map(0, &range, &data));
        memcpy(data, values.data(), sizeof(T) * values.size()); buffers[slot]->Unmap(0, nullptr);
    }
    template<class T> std::vector<T> Read(U slot, U count) {
        uint64_t bytes = uint64_t(sizeof(T)) * count;
        Require(bytes <= uint64_t(strides[slot]) * counts[slot], "Bad readback size");
        auto readback = Buffer(bytes, D3D12_HEAP_TYPE_READBACK); Open();
        D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {buffers[slot].Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
        list->ResourceBarrier(1, &b); list->CopyBufferRegion(readback.Get(), 0, buffers[slot].Get(), 0, bytes);
        std::swap(b.Transition.StateBefore, b.Transition.StateAfter); list->ResourceBarrier(1, &b); Submit();
        void* data; D3D12_RANGE range{0, SIZE_T(bytes)};
        Check(readback->Map(0, &range, &data)); std::vector<T> result(count);
        memcpy(result.data(), data, SIZE_T(bytes)); D3D12_RANGE written{0, 0}; readback->Unmap(0, &written);
        return result;
    }
    void Run(const std::string& name, U threads) {
        if (!pipelines.count(name)) {
            std::ifstream file(directory / (name + ".dxil"), std::ios::binary);
            Require(bool(file), "Missing compiled shader");
            std::vector<char> code((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd{}; pd.pRootSignature = root.Get();
            pd.CS = {code.data(), code.size()};
            Check(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pipelines[name])));
        }
        U groups = (threads + 63) / 64; if (groups == 0) groups = 1;
        constants.groupsX = groups; Open();
        list->SetPipelineState(pipelines[name].Get()); list->SetComputeRootSignature(root.Get());
        ID3D12DescriptorHeap* heaps[] = {heap.Get()}; list->SetDescriptorHeaps(1, heaps);
        auto gpu = heap->GetGPUDescriptorHandleForHeapStart(); list->SetComputeRootDescriptorTable(0, gpu);
        gpu.ptr += uint64_t(12) * descriptorSize; list->SetComputeRootDescriptorTable(1, gpu);
        list->SetComputeRoot32BitConstants(2, 16, &constants, 0); list->Dispatch(groups, 1, 1);
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        list->ResourceBarrier(1, &barrier); Submit();
    }
    void SwapHistory() { std::swap(buffers[4], buffers[5]); UAV(4); UAV(5); }
    void Clear() { Run("ClearCS", constants.capacity); Run("ResetCountCS", 1); }
    U RequestAndCompact() {
        Run("BeginCS", constants.capacity); Run("ResetCountCS", 1);
        Run("RequestCS", constants.requests); Run("CompactCS", constants.capacity);
        Run("ArgumentsCS", 1); return Read<U>(8, 1)[0];
    }
    void Shade(U count) {
        Run("MaterialCS", count); Run("LightingCS", count); Run("ResolveCS", count);
    }
    void SetLighting(float direct, bool valid = true) {
        Upload(13, std::vector<Material>(64, {.5f,.5f,.5f,valid ? 1u : 0u,.25f,.25f,.25f,0}));
        Upload(14, std::vector<F4>(64, {direct,direct,direct,0}));
        Upload(15, std::vector<F4>(64, {0,0,0,0}));
    }
};

int wmain(int argc, wchar_t** argv) {
    try {
        Require(argc == 2, "Usage: Runtime.exe <compiled shader directory>");
        TestGPU gpu(argv[1]);
        std::vector<Surface> hits(1024, Hit(.25f));
        hits[0] = Hit(-131072.f); hits[1] = Hit(131071.f);
        hits[2] = Hit(-131073.f); hits[3] = Hit(131072.f); hits[4] = Hit(.25f, -1.f);
        gpu.Upload(12, hits); gpu.Run("KeyCS", 1024);
        auto keys = gpu.Read<U4>(11, 1024);
        Require(keys[0].w && keys[1].w && !keys[2].w && !keys[3].w, "Coordinate bounds alias");
        Require(keys[4].y != keys[5].y, "Normal bins alias");
        std::cout << "PASS exact key bounds and normal bins\n";

        for (U i = 0; i < 1024; ++i) hits[i] = Hit(.1f + float(i) / 2048.f);
        gpu.Upload(12, hits); gpu.SetLighting(3.14159265359f); gpu.Clear();
        Require(gpu.RequestAndCompact() == 1, "Concurrent duplicate requests did not merge");
        auto indices = gpu.Read<U>(10, 1024); U entry = indices[0];
        for (U i : indices) Require(i == entry, "Same-cell requests disagree");
        auto surface = gpu.Read<Surface>(2, 64)[entry];
        Require(surface.x >= .1f && surface.x < .6f && Near(surface.nz, 1), "Invalid representative");
        gpu.Shade(1); gpu.Run("GatherCS", 1024);
        for (auto r : gpu.Read<F4>(11, 1024)) Require(Near(r.x, .75f) && Near(r.y, .75f) && Near(r.z, .75f) && r.w == 1, "Bad diffuse radiance/gather");
        std::cout << "PASS concurrent request deduplication, representative and diffuse gather\n";

        gpu.SwapHistory(); gpu.constants.frame = 1; gpu.SetLighting(3 * 3.14159265359f);
        Require(gpu.RequestAndCompact() == 1, "Entry lost across frames");
        gpu.Run("LookupCS", 1024);
        for (auto r : gpu.Read<F4>(11, 1024)) Require(Near(r.x, .75f) && r.w == 1, "History was not frozen");
        gpu.Shade(1); gpu.Run("GatherCS", 1024);
        for (auto r : gpu.Read<F4>(11, 1024)) Require(Near(r.x, 1.25f) && r.w == 1, "Temporal average is wrong");
        Require(Near(gpu.Read<F4>(4, 64)[entry].x, .75f), "Resolve modified previous history");
        std::cout << "PASS frozen history and temporal resolve\n";

        gpu.SwapHistory(); gpu.constants.frame = 2; gpu.SetLighting(1, false);
        gpu.RequestAndCompact(); gpu.Shade(1); gpu.Run("GatherCS", 1024);
        for (auto r : gpu.Read<F4>(11, 1024)) Require(r.w == 0, "Missing material returned stale lighting");
        gpu.SwapHistory(); gpu.constants.frame = 3;
        gpu.SetLighting(std::numeric_limits<float>::quiet_NaN());
        gpu.RequestAndCompact(); gpu.Shade(1); gpu.Run("GatherCS", 1024);
        for (auto r : gpu.Read<F4>(11, 1024)) Require(r.w == 0, "Nonfinite estimate was accepted");
        std::cout << "PASS invalid material and nonfinite estimate rejection\n";

        // Two colliding keys, with the first deleted while the second survives.
        gpu.constants = Constants{}; gpu.constants.capacity = 2; gpu.constants.stale = 1;
        for (U i = 0; i < 1024; ++i) hits[i] = Hit(float(i) + .25f);
        gpu.Upload(12, hits); gpu.Run("KeyCS", 1024); keys = gpu.Read<U4>(11, 1024);
        U match = 1; while (match < 1024 && keys[match].z != keys[0].z) ++match;
        Require(match < 1024, "No test collision found");
        Surface a = hits[0], b = hits[match];
        hits[0] = a; gpu.Upload(12, hits); gpu.constants.requests = 1; gpu.Clear();
        gpu.Run("BeginCS", 2); gpu.Run("RequestCS", 1); // Force A into the first slot.
        U first = gpu.Read<U>(10, 1)[0]; hits[0] = b; gpu.Upload(12, hits); gpu.Run("RequestCS", 1);
        U second = gpu.Read<U>(10, 1)[0]; Require(first != second && second < 2, "Collision insertion failed");
        gpu.Run("CompactCS", 2); gpu.SetLighting(3.14159265359f); gpu.Shade(2);
        gpu.SwapHistory(); gpu.constants.frame = 1;
        Require(gpu.RequestAndCompact() == 2, "Retained unrequested entry was not compacted"); gpu.Shade(2);
        gpu.SwapHistory(); gpu.constants.frame = 2;
        Require(gpu.RequestAndCompact() == 1, "Deletion created a duplicate past a hole");
        Require(gpu.Read<U>(10, 1)[0] == second, "Lookup did not search past deleted slot");
        Require(gpu.Read<F4>(4, 2)[first].w == 0, "Evicted history was not cleared");
        std::cout << "PASS retained updates, aging and lookup past a deleted collision\n";

        gpu.constants = Constants{}; gpu.constants.capacity = 1; gpu.constants.requests = 2;
        hits[0] = a; hits[1] = b; gpu.Upload(12, hits); gpu.Clear(); gpu.SetLighting(3.14159265359f);
        Require(gpu.RequestAndCompact() == 1, "Capacity limit failed"); indices = gpu.Read<U>(10, 2);
        Require((indices[0] == 0 && indices[1] == 0xffffffffu) ||
            (indices[1] == 0 && indices[0] == 0xffffffffu), "Full probe window overwrote data");
        gpu.Shade(1); gpu.SwapHistory(); gpu.constants.frame = 17; gpu.constants.requests = 0;
        Require(gpu.RequestAndCompact() == 0, "Stale entry not evicted"); gpu.Shade(0);
        auto args = gpu.Read<U>(9, 3); Require(args[0] == 1 && args[1] == 1 && args[2] == 1, "Bad empty indirect args");
        gpu.SwapHistory(); gpu.constants.frame = 18; gpu.constants.requests = 1;
        Require(gpu.RequestAndCompact() == 1, "Slot could not be reused");
        gpu.SetLighting(3 * 3.14159265359f); gpu.Shade(1); gpu.Run("GatherCS", 1);
        auto result = gpu.Read<F4>(11, 1)[0]; Require(Near(result.x, 1.75f) && result.w == 1, "Slot reuse inherited old history");
        std::cout << "PASS full table failure, empty dispatch and clean slot reuse\n";
        std::cout << "All DSHARC D3D12 runtime checks passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
