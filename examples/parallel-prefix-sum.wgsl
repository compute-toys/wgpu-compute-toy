// A simple demo of inclusive scan, using workgroup shared memory. Put some values into the Input array, and Result[i] will give you the sum Input[0] + Input[1] + ... Input[i]

#define SCAN_TYPE float
#define SCAN_WORKGROUP_SIZE 256
#include "davidar/scan"

#define INPUT_SIZE 2048
// BUCKET_COUNT = INPUT_SIZE / SCAN_WORKGROUP_SIZE
#define BUCKET_COUNT 8

struct Store {
    Input: array<SCAN_TYPE, INPUT_SIZE>,
    Result: array<SCAN_TYPE, INPUT_SIZE>,
    Auxiliary: array<SCAN_TYPE, BUCKET_COUNT>,
}

#storage store Store

#include "Dave_Hoskins/hash"

fn GenerateInput(i: uint) {
    store.Input[i] = pow(hash12(float2(uint2(i, time.frame / 30u))), 9.);
}

// scan in each bucket
@compute @workgroup_size(SCAN_WORKGROUP_SIZE)
#workgroup_count ScanInBucket BUCKET_COUNT 1 1
fn ScanInBucket(
    @builtin(global_invocation_id) DTid: uint3,
    @builtin(local_invocation_index) GI: uint,
) {
    GenerateInput(DTid.x);
    store.Result[DTid.x] = scan_pass(GI, store.Input[DTid.x]);
}

// record and scan the sum of each bucket
@compute @workgroup_size(BUCKET_COUNT)
#workgroup_count ScanBucketResult 1 1 1
fn ScanBucketResult(
    @builtin(global_invocation_id) DTid: uint3,
    @builtin(local_invocation_index) GI: uint,
) {
    store.Auxiliary[DTid.x] = scan_pass(GI,
        select(SCAN_TYPE(0), store.Result[DTid.x * uint(SCAN_WORKGROUP_SIZE - 1)], DTid.x > 0u));
}

// add the bucket scanned result to each bucket to get the final result
@compute @workgroup_size(SCAN_WORKGROUP_SIZE)
#workgroup_count ScanAddBucketResult BUCKET_COUNT 1 1
fn ScanAddBucketResult(
    @builtin(workgroup_id) Gid: uint3,
    @builtin(global_invocation_id) DTid: uint3,
) {
    store.Result[DTid.x] += store.Auxiliary[Gid.x];
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    var col = float3(0.);
    let u = float(screen_size.y - id.y);
    if (id.x < uint(INPUT_SIZE)) {
        if (float(id.y) < store.Input[id.x] * float(screen_size.y / 10u)) {
            col = float3(.1);
        }
        let y = store.Result[id.x];
        if (u < 5. * y) {
            col = float3(1.);
        }
        let a = store.Auxiliary[id.x / uint(SCAN_WORKGROUP_SIZE)];
        if (u < 5. * a) {
            col = float3(.5);
        }
    }
    //assert(0, store.Result[1] == store.Input[0] + store.Input[1]);

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
