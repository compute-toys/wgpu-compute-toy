/* Partial port of PBRT v3 <https://github.com/mmp/pbrt-v3> to WGSL
   with support for image based lighting. Quite slow to compile and still a bit buggy.
   https://pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Path_Tracing

BSD 2-Clause License

Copyright (c) 1998-2015, Matt Pharr, Greg Humphreys, and Wenzel Jakob
Copyright (c) 2022, David A Roberts <https://davidar.io/>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

struct BSDF {
    Rd: float3,
    Rs: float3,
    roughness: float,
}

fn isfinite(x: f32) -> bool {
    return clamp(x, -3.4e38, 3.4e38) == x;
}

// A.1 Main Include File

const Pi = 3.14159265358979323846;

// A.5 Mathematical Routines

fn Quadratic(a: f32, b: f32, c: f32, t0: ptr<function, f32>, t1: ptr<function, f32>) -> bool {
    // Find quadratic discriminant
    let discrim = b * b - 4. * a * c;
    if (discrim < 0.) { return false; }
    let rootDiscrim = sqrt(discrim);

    // Compute quadratic t values
    var q = 0.;
    if (b < 0.) {
        q = -.5 * (b - rootDiscrim);
    } else {
        q = -.5 * (b + rootDiscrim);
    }
    *t0 = q / a;
    *t1 = c / q;
    if (*t0 > *t1) {
        let swap = *t0;
        *t0 = *t1;
        *t1 = swap;
    }
    return true;
}

// 2.2 Vectors

alias Vector3f = vec3<f32>;

fn MinComponent(v: Vector3f) -> f32 {
    return min(v.x, min(v.y, v.z));
}

fn MaxComponent(v: Vector3f) -> f32 {
    return max(v.x, max(v.y, v.z));
}

fn MaxDimension(v: Vector3f) -> i32 {
    if (v.x > v.y) {
        if (v.x > v.z) {
            return 0;
        } else {
            return 2;
        }
    } else {
        if (v.y > v.z) {
            return 1;
        } else {
            return 2;
        }
    }
}

fn Permute(v: Vector3f, x: i32, y: i32, z: i32) -> Vector3f {
    return Vector3f(v[x], v[y], v[z]);
}

fn CoordinateSystem(v1: Vector3f, v2: ptr<function, Vector3f>, v3: ptr<function, Vector3f>) {
    if (abs(v1.x) > abs(v1.y)) {
        *v2 = Vector3f(-v1.z, 0., v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    } else {
        *v2 = Vector3f(0., v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    }
    *v3 = cross(v1, *v2);
}

// 2.3 Points

alias Point2f = vec2<f32>;
alias Point3f = vec3<f32>;

// 2.4 Normals

alias Normal3f = vec3<f32>;

// 2.5 Rays

struct Ray {
    o: Point3f,
    d: Vector3f,
}

// 2.6 Bounding Boxes

struct Bounds2f {
    pMin: Point2f,
    pMax: Point2f,
}

struct Bounds3f {
    pMin: Point3f,
    pMax: Point3f,
}

// 2.7 Transformations

alias Matrix4x4 = mat4x4<f32>;

struct Transform {
    m: Matrix4x4,
    mInv: Matrix4x4,
}

fn Translate(delta: Vector3f) -> Transform {
    let m = transpose(Matrix4x4(
        1., 0., 0., delta.x,
        0., 1., 0., delta.y,
        0., 0., 1., delta.z,
        0., 0., 0.,      1.,
    ));
    let mInv = transpose(Matrix4x4(
        1., 0., 0., -delta.x,
        0., 1., 0., -delta.y,
        0., 0., 1., -delta.z,
        0., 0., 0.,       1.,
    ));
    return Transform(m, mInv);
}

fn Scale(x: f32, y: f32, z: f32) -> Transform {
    let m = transpose(Matrix4x4(
         x, 0., 0., 0.,
        0.,  y, 0., 0.,
        0., 0.,  z, 0.,
        0., 0., 0., 1.));
    let mInv = transpose(Matrix4x4(
        1./x,   0.,   0., 0.,
        0.,   1./y,   0., 0.,
        0.,     0., 1./z, 0.,
        0.,     0.,   0., 1.));
    return Transform(m, mInv);
}

fn RotateX(theta: f32) -> Transform {
    let sinTheta = sin(radians(theta));
    let cosTheta = cos(radians(theta));
    let m = transpose(Matrix4x4(
        1.,       0.,        0., 0., 
        0., cosTheta, -sinTheta, 0.,
        0., sinTheta,  cosTheta, 0.,
        0.,       0.,        0., 1.));
    return Transform(m, transpose(m));
}

fn RotateY(theta: f32) -> Transform {
    let sinTheta = sin(radians(theta));
    let cosTheta = cos(radians(theta));
    let m = transpose(Matrix4x4(
        cosTheta,  0.,  sinTheta, 0., 
        0.,        1.,        0., 0.,
        -sinTheta, 0.,  cosTheta, 0.,
        0.,        0.,        0., 1.));
    return Transform(m, transpose(m));
}

fn RotateZ(theta: f32) -> Transform {
    let sinTheta = sin(radians(theta));
    let cosTheta = cos(radians(theta));
    let m = transpose(Matrix4x4(
        cosTheta, -sinTheta, 0., 0.,
        sinTheta,  cosTheta, 0., 0.,
        0.,       0.,        1., 0., 
        0.,       0.,        0., 1.));
    return Transform(m, transpose(m));
}

fn Transform_Inverse(t: Transform) -> Transform {
    return Transform(t.mInv, t.m);
}

// 2.8 Applying Transformations

fn Transform_Point3f(t: Transform, p: Point3f) -> Point3f {
    let q = t.m * vec4<f32>(p, 1.);
    return q.xyz / q.w;
}

fn Transform_Vector3f(t: Transform, v: Vector3f) -> Vector3f {
    return (t.m * vec4<f32>(v, 0.)).xyz;
}

fn Transform_Normal3f(t: Transform, n: Normal3f) -> Normal3f {
    return (transpose(t.mInv) * vec4<f32>(n, 0.)).xyz;
}

fn Transform_Ray(t: Transform, r: Ray) -> Ray {
    let o = Transform_Point3f(t, r.o);
    let d = Transform_Vector3f(t, r.d);
    return Ray(o, d);
}

fn Compose(t: Transform, t2: Transform) -> Transform {
    return Transform(t.m * t2.m, t2.mInv * t.mInv);
}

fn Transform_SwapsHandedness(t: Transform) -> bool {
    let m = mat3x3<f32>(t.m[0].xyz, t.m[1].xyz, t.m[2].xyz);
    return determinant(m) < 0.;
}

// 2.10 Interactions

struct TangentSpace {
    n: Normal3f,
    dpdu: Vector3f,
    dpdv: Vector3f,
}

struct SurfaceInteraction {
    p: Point3f,
    wo: Vector3f,

    uv: Point2f,
    t: TangentSpace,
    shading: TangentSpace,
}

fn Transform_TangentSpace(t: Transform, s: TangentSpace) -> TangentSpace {
    return TangentSpace(
        normalize(Transform_Normal3f(t, s.n)),
        Transform_Vector3f(t, s.dpdu),
        Transform_Vector3f(t, s.dpdv));
}

fn Transform_SurfaceInteraction(t: Transform, si: SurfaceInteraction) -> SurfaceInteraction {
    return SurfaceInteraction(
        Transform_Point3f(t, si.p),
        Transform_Vector3f(t, si.wo),
        si.uv,
        Transform_TangentSpace(t, si.t),
        Transform_TangentSpace(t, si.shading));
}

// 3.2 Spheres

struct Sphere {
    ObjectToWorld: Transform,
    radius: f32,
    zMin: f32,
    zMax: f32,
    phiMax: f32,
}

fn Sphere_ObjectBound(s: Sphere) -> Bounds3f {
    return Bounds3f(Point3f(-s.radius, -s.radius, s.zMin),
                    Point3f( s.radius,  s.radius, s.zMax));
}

fn Sphere_Intersect(s: Sphere, r: Ray, isect: ptr<function, SurfaceInteraction>) -> bool {
    var phi = 0.;
    var pHit = Point3f(0.);

    // Transform Ray to object space
    let ray = Transform_Ray(Transform_Inverse(s.ObjectToWorld), r);

    // Compute quadratic sphere coefficients
    let o = ray.o;
    let d = ray.d;
    let a = dot(d,d);
    let b = 2. * dot(d,o);
    let c = dot(o,o) - s.radius * s.radius;

    // Solve quadratic equation for t values
    var t0 = 0.;
    var t1 = 0.;
    if (!Quadratic(a, b, c, &t0, &t1)) {
        return false;
    }

    // Check quadric shape t0 and t1 for nearest intersection
    if (/* t0 > ray.tMax || */ t1 <= 0.) {
        return false;
    }
    var tShapeHit = t0;
    if (tShapeHit <= 0.) {
        tShapeHit = t1;
        //if (tShapeHit > ray.tMax) { return false; }
    }

    // Compute sphere hit position and phi
    pHit = ray.o + ray.d * tShapeHit;
    // Refine sphere intersection point
    pHit *= s.radius / length(pHit);
    if (pHit.x == 0. && pHit.y == 0.) { pHit.x = 1e-5 * s.radius; }
    phi = atan2(pHit.y, pHit.x);
    if (phi < 0.) {
        phi += 2. * Pi;
    }

    // Test sphere intersection against clipping parameters
    if ((s.zMin > -s.radius && pHit.z < s.zMin) ||
        (s.zMax <  s.radius && pHit.z > s.zMax) || phi > s.phiMax) {
        if (tShapeHit == t1) { return false; }
        //if (t1 > ray.tMax) { return false; }
        tShapeHit = t1;

        // Compute sphere hit position and phi
        pHit = ray.o + ray.d * tShapeHit;
        // Refine sphere intersection point
        pHit *= s.radius / length(pHit);
        if (pHit.x == 0. && pHit.y == 0.) { pHit.x = 1e-5 * s.radius; }
        phi = atan2(pHit.y, pHit.x);
        if (phi < 0.) {
            phi += 2. * Pi;
        }

        if ((s.zMin > -s.radius && pHit.z < s.zMin) ||
            (s.zMax <  s.radius && pHit.z > s.zMax) || phi > s.phiMax) {
            return false;
        }
    }

    // Find parametric representation of sphere hit
    let u = phi / s.phiMax;
    let theta    = acos(clamp(pHit.z / s.radius, -1., 1.));
    let thetaMin = acos(clamp(s.zMin / s.radius, -1., 1.));
    let thetaMax = acos(clamp(s.zMax / s.radius, -1., 1.));
    let v = (theta - thetaMin) / (thetaMax - thetaMin);

    // Compute sphere dpdu and dpdv
    let zRadius = length(pHit.xy);
    let invZRadius = 1. / zRadius;
    let cosPhi = pHit.x * invZRadius;
    let sinPhi = pHit.y * invZRadius;
    let dpdu = Vector3f(-s.phiMax * pHit.y, s.phiMax * pHit.x, 0.);
    let dpdv = (thetaMax - thetaMin) *
        Vector3f(pHit.z * cosPhi, pHit.z * sinPhi,
                 -s.radius * sin(theta));

    // Initialize SurfaceInteraction from parametric information
    let n = normalize(cross(dpdu, dpdv));
    let t = TangentSpace(n, dpdu, dpdv);
    *isect = Transform_SurfaceInteraction(s.ObjectToWorld, SurfaceInteraction(pHit, -ray.d, Point2f(u,v), t, t));

    // Update tHit for quadric intersection
    //*tHit = tShapeHit;

    return true;
}

fn Sphere_Area(s: Sphere) -> f32 {
    return s.phiMax * s.radius * (s.zMax - s.zMin);
}

// 3.6 Triangle Meshes

struct Triangle {
    ObjectToWorld: Transform,
    p: array<Point3f,3>,
    uv: array<Point2f,3>,
}

fn Triangle_Intersect(tri: Triangle, ray: Ray, isect: ptr<function, SurfaceInteraction>) -> bool {
    // Get triangle vertices in p0, p1, and p2
    let p0 = tri.p[0];
    let p1 = tri.p[1];
    let p2 = tri.p[2];
    let uv = tri.uv;

    // Perform ray–triangle intersection test
    // Transform triangle vertices to ray coordinate space
    // Translate vertices based on ray origin
    var p0t = p0 - Vector3f(ray.o);
    var p1t = p1 - Vector3f(ray.o);
    var p2t = p2 - Vector3f(ray.o);

    // Permute components of triangle vertices and ray direction
    let kz = MaxDimension(abs(ray.d));
    var kx = kz + 1; if (kx == 3) { kx = 0; }
    var ky = kx + 1; if (ky == 3) { ky = 0; }
    let d = Permute(ray.d, kx, ky, kz);
    p0t = Permute(p0t, kx, ky, kz);
    p1t = Permute(p1t, kx, ky, kz);
    p2t = Permute(p2t, kx, ky, kz);

    // Apply shear transformation to translated vertex positions
    let Sx = -d.x / d.z;
    let Sy = -d.y / d.z;
    let Sz = 1. / d.z;
    p0t.x += Sx * p0t.z;
    p0t.y += Sy * p0t.z;
    p1t.x += Sx * p1t.z;
    p1t.y += Sy * p1t.z;
    p2t.x += Sx * p2t.z;
    p2t.y += Sy * p2t.z;

    // Compute edge function coefficients e0, e1, and e2
    let e0 = p1t.x * p2t.y - p1t.y * p2t.x;
    let e1 = p2t.x * p0t.y - p2t.y * p0t.x;
    let e2 = p0t.x * p1t.y - p0t.y * p1t.x;

    // TODO: Fall back to double-precision test at triangle edges

    // Perform triangle edge and determinant tests
    if ((e0 < 0. || e1 < 0. || e2 < 0.) && (e0 > 0. || e1 > 0. || e2 > 0.)) {
        return false;
    }
    let det = e0 + e1 + e2;
    if (det == 0.) {
        return false;
    }

    // Compute scaled hit distance to triangle and test against ray t range
    p0t.z *= Sz;
    p1t.z *= Sz;
    p2t.z *= Sz;
    let tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
    if (det < 0. && (tScaled >= 0. /* || tScaled < ray.tMax * det */)) {
        return false;
    } else if (det > 0. && (tScaled <= 0. /* || tScaled > ray.tMax * det */)) {
        return false;
    }

    // Compute barycentric coordinates and t value for triangle intersection
    let invDet = 1. / det;
    let b0 = e0 * invDet;
    let b1 = e1 * invDet;
    let b2 = e2 * invDet;
    let t = tScaled * invDet;

    // TODO: Ensure that computed triangle t is conservatively greater than zero

    // Compute triangle partial derivatives
    var dpdu = Vector3f(0.);
    var dpdv = Vector3f(0.);
    // Compute deltas for triangle partial derivatives
    let duv02 = uv[0] - uv[2];
    let duv12 = uv[1] - uv[2];
    let dp02 = p0 - p2;
    let dp12 = p1 - p2;

    let determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];
    if (determinant == 0.) {
        // Handle zero determinant for triangle partial derivative matrix
        CoordinateSystem(normalize(cross(p2 - p0, p1 - p0)), &dpdu, &dpdv);
    } else {
        let invdet = 1. / determinant;
        dpdu = ( duv12[1] * dp02 - duv02[1] * dp12) * invdet;
        dpdv = (-duv12[0] * dp02 + duv02[0] * dp12) * invdet;
    }

    // TODO: Compute error bounds for triangle intersection

    // Interpolate uv parametric coordinates and hit point
    let pHit = b0 * p0 + b1 * p1 + b2 * p2;
    let uvHit = b0 * uv[0] + b1 * uv[1] + b2 * uv[2];

    // TODO: Test intersection against alpha texture, if present

    // Fill in SurfaceInteraction from triangle hit
    let n = Normal3f(normalize(cross(dp02, dp12)));
    let ts = TangentSpace(n, dpdu, dpdv);
    *isect = SurfaceInteraction(pHit, -ray.d, uvHit, ts, ts);

    // TODO: Initialize Triangle shading geometry
    // TODO: Ensure correct orientation of the geometric normal

    //*tHit = t;
    return true;
}

fn Quad_Intersect(ray: Ray, isect: ptr<function, SurfaceInteraction>, a: Point3f, b: Point3f, c: Point3f, d: Point3f) -> bool {
    let Identity = Matrix4x4(
        1., 0., 0., 0.,
        0., 1., 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 1.,
    );
    let uv = array<Point2f,3>(Point2f(0., 0.), Point2f(1., 0.), Point2f(1., 1.));
    let t = Transform(Identity, Identity);
    return Triangle_Intersect(Triangle(t, array<Point3f,3>(a,b,c), uv), ray, isect)
        || Triangle_Intersect(Triangle(t, array<Point3f,3>(c,d,a), uv), ray, isect);
}

// 3.9 Managing Rounding Error

fn SurfaceInteraction_SpawnRay(si: SurfaceInteraction, d: Vector3f) -> Ray {
    return Ray(si.p + si.t.n * 1e-3, d);
}

// 5.1 Spectral Representation

alias RGBSpectrum = vec3<f32>;
alias Spectrum = RGBSpectrum;

fn Spectrum_IsBlack(f: Spectrum) -> bool {
    return f.r == 0. && f.g == 0. && f.b == 0.;
}

fn Spectrum_y(f: Spectrum) -> float {
    // TODO: compute y coefficient of XYZ colour
    return MaxComponent(f);
}

// 5.5 Working with Radiometric Integrals

fn SphericalDirection(sinTheta: f32, cosTheta: f32, phi: f32) -> Vector3f {
    return Vector3f(sinTheta * cos(phi),
                    sinTheta * sin(phi),
                    cosTheta);
}

fn SphericalTheta(v: Vector3f) -> f32 {
    return acos(clamp(v.z, -1., 1.));
}

fn SphericalPhi(v: Vector3f) -> f32 {
    let p = atan2(v.y, v.x);
    return select(p, p + 2. * Pi, p < 0.);
}

// 7.2 Sampling Interface

alias Sampler = ptr<function, u32>;

fn pcg_random(seed: Sampler) -> f32 {
    *seed = *seed * 747796405u + 2891336453u;
    let word = ((*seed >> ((*seed >> 28u) + 4u)) ^ *seed) * 277803737u;
    return f32((word >> 22u) ^ word) / f32(0xffffffffu);
}

fn Sampler_Get1D(samp: Sampler) -> f32 {
    return pcg_random(samp);
}

fn Sampler_Get2D(samp: Sampler) -> Point2f {
    return Point2f(Sampler_Get1D(samp), Sampler_Get1D(samp));
}

// 8 Reflection Models

fn CosTheta(w: Vector3f) -> f32 { return w.z; }
fn Cos2Theta(w: Vector3f) -> f32 { return w.z * w.z; }
fn AbsCosTheta(w: Vector3f) -> f32 { return abs(w.z); }

fn Sin2Theta(w: Vector3f) -> f32 {
    return max(0., 1. - Cos2Theta(w));
}
fn SinTheta(w: Vector3f) -> f32 {
    return sqrt(Sin2Theta(w));
}

fn TanTheta(w: Vector3f) -> f32 {
    return SinTheta(w) / CosTheta(w);
}
fn Tan2Theta(w: Vector3f) -> f32 {
    return Sin2Theta(w) / Cos2Theta(w);
}

fn CosPhi(w: Vector3f) -> f32 {
    let sinTheta = SinTheta(w);
    if (sinTheta == 0.) {
        return 1.;
    } else {
        return clamp(w.x / sinTheta, -1., 1.);
    }
}
fn SinPhi(w: Vector3f) -> f32 {
    let sinTheta = SinTheta(w);
    if (sinTheta == 0.) {
        return 0.;
    } else {
        return clamp(w.y / sinTheta, -1., 1.);
    }
}

fn Cos2Phi(w: Vector3f) -> f32 {
    return CosPhi(w) * CosPhi(w);
}
fn Sin2Phi(w: Vector3f) -> f32 {
    return SinPhi(w) * SinPhi(w);
}

fn CosDPhi(wa: Vector3f, wb: Vector3f) -> f32 {
    return clamp(dot(wa,wb) / (length(wa) * length(wb)), -1., 1.);
}

// 8.2 Specular Reflection and Transmission

fn Reflect(wo: Vector3f, n: Vector3f) -> Vector3f {
    //return -wo + 2. * dot(wo, n) * n;
    return -reflect(wo, n);
}

// 8.4 Microfacet Models

fn BeckmannDistribution_RoughnessToAlpha(roughness: f32) -> f32 {
    return pow(max(roughness, 1e-3), 2.);
    //let x = log(max(roughness, 1e-3));
    //return 1.62142 + 0.819955 * x + 0.1734 * x * x +
    //       0.0171201 * x * x * x + 0.000640711 * x * x * x * x;
}

fn BeckmannDistribution_D(alpha: f32, wh: Vector3f) -> f32 {
    let tan2Theta = Tan2Theta(wh);
    if (!isfinite(tan2Theta)) { return 0.; }
    let cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
    return exp(-tan2Theta * (Cos2Phi(wh) / (alpha * alpha) +
                             Sin2Phi(wh) / (alpha * alpha))) /
        (Pi * alpha * alpha * cos4Theta);
}

// 8.5 Fresnel Incidence Effects

fn SchlickFresnel(Rs: Spectrum, cosTheta: f32) -> Spectrum {
    return Rs + pow(1. - cosTheta, 5.) * (Spectrum(1.) - Rs);
}

fn FresnelBlend_f(bsdf: BSDF, wo: Vector3f, wi: Vector3f) -> Spectrum {
    let alpha = BeckmannDistribution_RoughnessToAlpha(bsdf.roughness);

    let diffuse = (28./(23.*Pi)) * bsdf.Rd *
        (Spectrum(1.) - bsdf.Rs) *
        (1. - pow(1. - .5 * AbsCosTheta(wi), 5.)) *
        (1. - pow(1. - .5 * AbsCosTheta(wo), 5.));
    var wh = wi + wo;
    if (wh.x == 0. && wh.y == 0. && wh.z == 0.) { return Spectrum(0.); }
    wh = normalize(wh);
    let specular = 1. / //BeckmannDistribution_D(alpha, wh) /
        (4. * abs(dot(wi, wh)) *
         max(AbsCosTheta(wi), AbsCosTheta(wo))) *
         SchlickFresnel(bsdf.Rs, dot(wi, wh));
    return diffuse + specular;
}

// 9.1 BSDFs

fn BSDF_WorldToLocal(si: SurfaceInteraction, v: Vector3f) -> Vector3f {
    let ns = si.shading.n;
    let ss = normalize(si.shading.dpdu);
    let ts = cross(ns, ss);
    return Vector3f(dot(v, ss), dot(v, ts), dot(v, ns));
}

fn BSDF_LocalToWorld(si: SurfaceInteraction, v: Vector3f) -> Vector3f {
    let ns = si.shading.n;
    let ss = normalize(si.shading.dpdu);
    let ts = cross(ns, ss);
    return Vector3f(ss.x * v.x + ts.x * v.y + ns.x * v.z,
                    ss.y * v.x + ts.y * v.y + ns.y * v.z,
                    ss.z * v.x + ts.z * v.y + ns.z * v.z);
}

// 12.6 Infinite Area Lights

fn SurfaceInteraction_Le(isect: SurfaceInteraction, w: float3) -> Spectrum {
    return Spectrum(0.);
}

fn InfiniteAreaLight_Le(LightToWorld: Transform, ray: Ray) -> Spectrum {
    let w = normalize(Transform_Vector3f(Transform_Inverse(LightToWorld), ray.d));
    let st = Point2f(SphericalPhi(w) / (2.*Pi), SphericalTheta(w) / Pi);
    return Spectrum(textureSampleLevel(channel0, bilinear, st, 0.).rgb);
}

// 13.6 2D Sampling with Multidimensional Transformations

fn ConcentricSampleDisk(u: Point2f) -> Point2f {
    // Map uniform random numbers to [-1, 1]^2
    let uOffset = 2. * u - 1.;

    // Handle degeneracy at the origin
    if (uOffset.x == 0. && uOffset.y == 0.) {
        return Point2f(0.);
    }

    // Apply concentric mapping to point
    var theta = 0.;
    var r = 0.;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = Pi/4. * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = Pi/2. - Pi/4. * (uOffset.x / uOffset.y);
    }
    return r * Point2f(cos(theta), sin(theta));
}

fn InfiniteAreaLight_img(pos: int2, mip: int) -> float {
    // Compute scalar-valued image img from environment map
    return Spectrum_y(textureLoad(channel0, pos, mip).rgb);
}

fn Distribution2D_SampleContinuous(samp: Sampler, u: Point2f, pdf: ptr<function, f32>) -> Point2f {
    let mipmax = int(textureNumLevels(channel0)) - 1;
    let res = float2(textureDimensions(channel0));
    var prob = 1.;
    var pos = int2(0,0);
    for (var mip = mipmax - 1; mip >= 0; mip -= 1) {
        pos *= 2;
        let w00 = InfiniteAreaLight_img(pos + int2(0,0), mip);
        let w01 = InfiniteAreaLight_img(pos + int2(0,1), mip);
        let w10 = InfiniteAreaLight_img(pos + int2(1,0), mip);
        let w11 = InfiniteAreaLight_img(pos + int2(1,1), mip);
        let w0 = w00 + w01; // weight of column 0
        let w1 = w10 + w11; // weight of column 1
        let w = w0 + w1; // total weight
        let offset = select(
            int2(0, select(0, 1, Sampler_Get1D(samp) <= w01 / w0)), // cond prob of row 1 given col 0
            int2(1, select(0, 1, Sampler_Get1D(samp) <= w11 / w1)), // cond prob of row 1 given col 1
            Sampler_Get1D(samp) <= w1 / w); // prob of col 1
        pos += offset;
        prob *= select(
            w0 / w * select(w00 / w0, w01 / w0, offset.y == 1),
            w1 / w * select(w10 / w1, w11 / w1, offset.y == 1),
            offset.x == 1);
    }
    let uv = (float2(pos) + Sampler_Get2D(samp)) / res;
    *pdf = prob * res.x * res.y;
    return uv;
}

fn CosineSampleHemisphere(u: Point2f) -> Vector3f {
    let d = ConcentricSampleDisk(u);
    let z = sqrt(max(0., 1. - dot(d,d)));
    return Vector3f(d.xy, z);
}

// 13.10 Importance Sampling

fn PowerHeuristic(nf: i32, fPdf: f32, ng: i32, gPdf: f32) -> f32 {
    let f = f32(nf) * fPdf;
    let g = f32(ng) * gPdf;
    return (f * f) / (f * f + g * g);
}

// 14.1 Sampling Reflection Functions

fn SameHemisphere(w: Vector3f, wp: Vector3f) -> bool {
    return w.z * wp.z > 0.;
}

fn BeckmannDistribution_Sample_wh(alpha: f32, wo: Vector3f, u: Point2f) -> Vector3f {
    // Sample full distribution of normals for Beckmann distribution
    // Compute tan2Theta and phi for Beckmann distribution sample
    var logSample = log(1. - u[0]);
    if (!isfinite(logSample)) {
        logSample = 0.;
    }
    let tan2Theta = -alpha * alpha * logSample;
    let phi = u[1] * 2. * Pi;

    // TODO: Compute tan2Theta and phi for anisotropic Beckmann distribution

    // Map sampled Beckmann angles to normal direction wh>> 
    let cosTheta = 1. / sqrt(1. + tan2Theta);
    let sinTheta = sqrt(max(0., 1. - cosTheta * cosTheta));
    var wh = SphericalDirection(sinTheta, cosTheta, phi);
    if (!SameHemisphere(wo, wh)) {
        wh = -wh;
    }

    return wh;

    // TODO: Sample visible area of normals for Beckmann distribution>> 
}

fn BeckmannDistribution_Pdf(alpha: f32, wo: Vector3f, wh: Vector3f) -> f32 {
    //if (sampleVisibleArea)
    //    return D(wh) * G1(wo) * AbsDot(wo, wh) / AbsCosTheta(wo);
    //else
        return //BeckmannDistribution_D(alpha, wh) *
               AbsCosTheta(wh);
}

fn FresnelBlend_Pdf(bsdf: BSDF, wo: Vector3f, wi: Vector3f) -> f32 {
    let alpha = BeckmannDistribution_RoughnessToAlpha(bsdf.roughness);

    if (!SameHemisphere(wo, wi)) { return 0.; }
    let wh = normalize(wo + wi);
    let pdf_wh = BeckmannDistribution_Pdf(alpha, wo, wh);
    return .5 * (AbsCosTheta(wi) / Pi + pdf_wh / (4. * dot(wo, wh)));
}

fn FresnelBlend_Sample_f(bsdf: BSDF, wo: Vector3f, wi: ptr<function, Vector3f>, uOrig: Point2f, pdf: ptr<function, f32>) -> Spectrum {
    let alpha = BeckmannDistribution_RoughnessToAlpha(bsdf.roughness);

    var u = uOrig;
    if (u[0] < .5) { // LambertianReflection
        u[0] = 2. * u[0];
        // Cosine-sample the hemisphere, flipping the direction if necessary
        *wi = CosineSampleHemisphere(u);
        if (wo.z < 0.) {
            (*wi).z *= -1.;
        }
    } else { // MicrofacetReflection
        u[0] = 2. * (u[0] - .5);
        // Sample microfacet orientation wh and reflected direction wi
        let wh = BeckmannDistribution_Sample_wh(alpha, wo, u);
        *wi = Reflect(wo, wh);
        if (!SameHemisphere(wo, *wi)) {
            return Spectrum(0.);
        }
    }
    *pdf = FresnelBlend_Pdf(bsdf, wo, *wi);
    return FresnelBlend_f(bsdf, wo, *wi);
}

fn BSDF_Sample_f(si: SurfaceInteraction, bsdf: BSDF, woWorld: Vector3f, wiWorld: ptr<function, Vector3f>, u: Point2f, pdf: ptr<function, f32>) -> Spectrum {
    // TODO: Choose which BxDF to sample
    // TODO: Remap BxDF sample u

    // Sample chosen BxDF
    var wi = Vector3f(0.);
    let wo = BSDF_WorldToLocal(si, woWorld);
    *pdf = 0.;
    let f = FresnelBlend_Sample_f(bsdf, wo, &wi, u, pdf);
    if (*pdf == 0.) {
        return Spectrum(0.);
    }
    *wiWorld = BSDF_LocalToWorld(si, wi);

    // TODO: Compute overall PDF with all matching BxDFs
    // TODO: Compute value of BSDF for sampled direction

    return f;
}

// 14.2 Sampling Light Sources

fn InfiniteAreaLight_Sample_Li(
    LightToWorld: Transform,
    si: SurfaceInteraction,
    u: Point2f,
    wi: ptr<function, Vector3f>,
    pdf: ptr<function, f32>,
    vis_p: ptr<function, Point3f>,
    samp: Sampler)
    -> Spectrum
{
    // Find uv sample coordinates in infinite light texture
    var mapPdf = 0.;
    let uv = Distribution2D_SampleContinuous(samp, u, &mapPdf);
    if (mapPdf == 0.) { return Spectrum(0.); }

    // Convert infinite light sample point to direction
    let theta = uv[1] * Pi;
    let phi = uv[0] * 2. * Pi;
    let cosTheta = cos(theta);
    let sinTheta = sin(theta);
    let sinPhi = sin(phi);
    let cosPhi = cos(phi);
    *wi = Transform_Vector3f(LightToWorld, Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));

    // Compute PDF for sampled infinite light direction
    *pdf = mapPdf / (2. * Pi * Pi * sinTheta);
    if (sinTheta == 0.) { *pdf = 0.; }

    // Return radiance value for infinite light direction
    let worldRadius = 1e3; // TODO
    *vis_p = si.p + *wi * (2. * worldRadius);
    return Spectrum(textureSampleLevel(channel0, bilinear, uv, 0.).rgb);
}

// Hardcoded Scene

fn Scene_Intersect(ray: Ray, isect: ptr<function, SurfaceInteraction>, bsdf: ptr<function, BSDF>) -> bool {
    var foundIntersection = false;

    let A = float3(-12.5, 25., -12.5);
    let B = float3( 12.5, 25., -12.5);
    let C = float3( 12.5, 25.,  12.5);
    let D = float3(-12.5, 25.,  12.5);
    let a = float3(-12.5, 15., -12.5);
    let b = float3( 12.5, 15., -12.5);
    let c = float3( 12.5, 15.,  12.5);
    let d = float3(-12.5, 15.,  12.5);

    // back wall
    if (Quad_Intersect(ray, isect, A, B, C, D)) {
        *bsdf = BSDF(float3(1.), float3(0.), 1.);
        foundIntersection = true;
    }

    // floor
    if (Quad_Intersect(ray, isect, a, b, B, A)) {
        *bsdf = BSDF(float3(1.), float3(0.), 1.);
        foundIntersection = true;
    }

    // ceiling
    if (Quad_Intersect(ray, isect, D, C, c, d) && !Quad_Intersect(ray, isect,
            float3(-5.0, 22.5, 12.5),
            float3( 5.0, 22.5, 12.5),
            float3( 5.0, 17.5, 12.5),
            float3(-5.0, 17.5, 12.5))) {
        *bsdf = BSDF(float3(1.), float3(0.), 1.);
        foundIntersection = true;
    }

    // left wall
    if (Quad_Intersect(ray, isect, a, A, D, d)) {
        *bsdf = BSDF(float3(1., 0., 0.), float3(0.), 1.);
        foundIntersection = true;
    }

    // right wall
    if (Quad_Intersect(ray, isect, B, b, c, C)) {
        *bsdf = BSDF(float3(0., 1., 0.), float3(0.), 1.);
        foundIntersection = true;
    }

    if (Sphere_Intersect(Sphere(Translate(float3(-9., 20., -9.)), 3., -1e6, 1e6, 1e6), ray, isect)) {
        *bsdf = BSDF(float3(.9, .9, .5), float3(.9, .9, .5), .2);
        foundIntersection = true;
    }

    if (Sphere_Intersect(Sphere(Translate(float3(0., 20., -9.)), 3., -1e6, 1e6, 1e6), ray, isect)) {
        *bsdf = BSDF(float3(.5, .9, .9), float3(.5, .9, .9), .2);
        foundIntersection = true;
    }

    if (Sphere_Intersect(Sphere(Translate(float3(9., 20., -9.)), 3., -1e6, 1e6, 1e6), ray, isect)) {
        *bsdf = BSDF(float3(0., 0., 1.), float3(1., 0., 0.), .4);
        foundIntersection = true;
    }

    for(var i = 0; i < 5; i += 1) {
        if (Sphere_Intersect(Sphere(Translate(float3(float(5*i - 10), 23., 0.)), 1.75, -1e6, 1e6, 1e6), ray, isect)) {
            *bsdf = BSDF(float3(.3, 1., .3), float3(.3, 1., .3), float(i*i) / 16.);
            foundIntersection = true;
        }
    }

    return foundIntersection;
}

fn Scene_IntersectP(ray: Ray) -> bool {
    var isect = SurfaceInteraction();
    var bsdf = BSDF();
    return Scene_Intersect(ray, &isect, &bsdf);
}

// 14.3 Direct Lighting

fn EstimateDirect(
    isect: SurfaceInteraction,
    bsdf: BSDF,
    uScattering: Point2f,
    LightToWorld: Transform,
    uLight: Point2f,
    samp: Sampler)
    -> Spectrum
{
    let specular = false;
    //let bsdfFlags = select(BSDF_ALL & ~BSDF_SPECULAR, BSDF_ALL, specular);
    var Ld = Spectrum(0.);
    // Sample light source with multiple importance sampling
    var wi = Vector3f(0.);
    var lightPdf = 0.;
    var scatteringPdf = 0.;
    var visibility_p = Point3f(0.);
    var Li = InfiniteAreaLight_Sample_Li(LightToWorld, isect, uLight, &wi, &lightPdf, &visibility_p, samp);
    if (lightPdf > 0. && !Spectrum_IsBlack(Li)) {
        // Compute BSDF or phase function's value for light sample
        // Evaluate BSDF for light sampling strategy
        let f = FresnelBlend_f(bsdf, isect.wo, wi) * abs(dot(wi, isect.shading.n));
        scatteringPdf = FresnelBlend_Pdf(bsdf, isect.wo, wi);

        // TODO: Evaluate phase function for light sampling strategy

        if (!Spectrum_IsBlack(f)) {
            // Compute effect of visibility for light source sample
            if (Scene_IntersectP(Ray(isect.p, visibility_p - isect.p))
                || dot(isect.t.n, visibility_p - isect.p) < 0.) {
                Li = Spectrum(0.);
            }

            // Add light's contribution to reflected radiance
            if (!Spectrum_IsBlack(Li)) {
                //if (IsDeltaLight(light.flags))
                //    Ld += f * Li / lightPdf;
                //else {
                    let weight = PowerHeuristic(1, lightPdf, 1, scatteringPdf);
                    Ld += f * Li * weight / lightPdf;
                //}
            }
        }
    }

    // TODO: Sample BSDF with multiple importance sampling

    return Ld;
}

fn UniformSampleOneLight(isect: SurfaceInteraction, bsdf: BSDF, LightToWorld: Transform, samp: Sampler, handleMedia: bool) -> Spectrum {
    // TODO: Randomly choose a single light to sample

    let uLight = Sampler_Get2D(samp);
    let uScattering = Sampler_Get2D(samp);
    let nLights = 1;
    return f32(nLights) * EstimateDirect(isect, bsdf, uScattering, LightToWorld, uLight, samp);
}

// 14.5 Path Tracing

fn PathIntegrator_Li(maxDepth: i32, r: Ray, samp: Sampler) -> Spectrum {
    let Identity = Matrix4x4(
        1., 0., 0., 0.,
        0., 1., 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 1.,
    );
    var L = Spectrum(0.);
    var beta = Spectrum(1.);
    var ray = r;
    var specularBounce = false;
    for (var bounces = 0; bounces <= maxDepth; bounces += 1) {
        // Find next path vertex and accumulate contribution
        // Intersect ray with scene and store intersection in isect
        var isect = SurfaceInteraction();
        var bsdf = BSDF();
        let foundIntersection = Scene_Intersect(ray, &isect, &bsdf);

        // Possibly add emitted light at intersection
        if (bounces == 0 || specularBounce) {
            // Add emitted light at path vertex or from the environment
            if (foundIntersection) {
                L += beta * SurfaceInteraction_Le(isect, -ray.d);
            } else {
                L += beta * InfiniteAreaLight_Le(Transform(Identity, Identity), ray);
            }
        }

        // Terminate path if ray escaped or maxDepth was reached
        if (!foundIntersection || bounces >= maxDepth) {
            break;
        }

        // TODO: Compute scattering functions and skip over medium boundaries

        // Sample illumination from lights to find path contribution
        L += beta * UniformSampleOneLight(isect, bsdf, Transform(Identity, Identity), samp, false);

        // Sample BSDF to get new path direction
        let wo = -ray.d;
        var wi = Vector3f(0.);
        var pdf = 0.;
        //var flags = BSDF_ALL;
        let f = BSDF_Sample_f(isect, bsdf, wo, &wi, Sampler_Get2D(samp), &pdf);
        if (Spectrum_IsBlack(f) || pdf == 0.) {
            break;
        }
        beta *= f * abs(dot(wi, isect.shading.n)) / pdf;
        //specularBounce = (flags & BSDF_SPECULAR) != 0u;
        ray = SurfaceInteraction_SpawnRay(isect, wi);

        // TODO: Account for subsurface scattering, if applicable

        // Possibly terminate the path with Russian roulette
        if (bounces > 3) {
            let q = max(.05, 1. - Spectrum_y(beta));
            if (Sampler_Get1D(samp) < q) {
                break;
            }
            beta /= max(1e-3, 1. - q);
        }
    }
    return L;
}

// Main Rendering

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn ACESFilm(x: float3) -> float3 {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x*(a*x + b)) / (x*(c*x + d) + e), float3(0.), float3(1.));
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));
    let resolution = float2(screen_size);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Seed PRNG
    var seed = id.x + id.y * screen_size.x + time.frame * screen_size.x * screen_size.y;

    // Camera
    let uv = (float2(id.xy) + Sampler_Get2D(&seed)) / resolution;
    var dir = Vector3f(uv.x * 2. - 1., 1., uv.y * 2. - 1.);
    dir.z /= -resolution.x / resolution.y;
    let ray = Ray(Point3f(0., -10., 0.), normalize(dir));

    // Path integration
    var col = float3(0.);
    let spp = 10; // samples per pixel (per frame)
    for (var i = 0; i < spp; i += 1) {
        col += PathIntegrator_Li(8, ray, &seed) / float(spp);
    }

    // Accumulate pixel samples
    col = mix(textureLoad(pass_in, int2(id.xy), 0, 0).rgb, col, 1. / float(time.frame + 1u));
    textureStore(pass_out, int2(id.xy), 0, float4(col, 1.));

    // Convert from HDR to LDR (tone mapping)
    col = ACESFilm(col);

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
