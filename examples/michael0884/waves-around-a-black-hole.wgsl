// Nonlinear waves in a Kerr-Newman metric

#define LENGTH 1024
#define WG 64

struct SimData
{
    data: array<array<array<vec4<f32>, LENGTH>, LENGTH>,2>,
}

#storage sim SimData
#define ITERATIONS 32
#define SIM_FRAME (uint(ITERATIONS)*time.frame+dispatch.id)
#define IN_BUF (SIM_FRAME%2u)
#define OUT_BUF ((SIM_FRAME+1u)%2u)

fn Load(i: uint2)  -> float4     { return sim.data[IN_BUF][i.x][i.y];  }
fn Store(i: uint2, data: float4) { sim.data[OUT_BUF][i.x][i.y] = data; }

fn sdSegment(p: float2, a: float2, b: float2) -> float
{
    let pa = p-a;
    let ba = b-a;
    let h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h );
}

fn sdBox(p: float2, b: float2) -> float
{
    let d = abs(p)-b;
    return length(max(d,float2(0.0))) + min(max(d.x,d.y),0.0);
}

alias Dual = vec2<f32>;

struct dVec
{
    a: array<Dual,3>,
}

fn dVecMake(t: Dual, x: Dual, y: Dual) -> dVec
{
    var v: dVec;
    v.a[0] = t;
    v.a[1] = x;
    v.a[2] = y;
    return v;
}

fn dC(a: float) -> Dual
{
    return Dual(a, 0.0);
}

fn dV(a: float) -> Dual
{
    return Dual(a, 1.0);
}

fn dMul(a: Dual, b: Dual) -> Dual
{
    return Dual(
        a.x * b.x,
        a.y * b.x + a.x * b.y,
    );
}

fn dChain(a: Dual, s: float, ds: float) -> Dual
{
    return Dual(
        s,
        ds * a.y,
    );
}

fn dSqrt(a: Dual) -> Dual
{
    let s = sqrt(abs(a.x));
    let ds = (0.5 / (s+0.0001));
    return dChain(a, s, ds);
}

fn dSin(a: Dual) -> Dual
{
   let s = sin(a.x);
   let ds = cos(a.x);
   return dChain(a, s, ds);
}

fn dCos(a: Dual) -> Dual
{
   let s = cos(a.x);
   let ds = -sin(a.x);
   return dChain(a, s, ds);
}

fn dSqr(a: Dual) -> Dual
{
   let s = a.x*a.x;
   let ds = 2.0*a.x;
   return dChain(a, s, ds);
}

fn dAbs(a: Dual) -> Dual
{
   let s = abs(a.x);
   let ds = sign(a.x);
   return dChain(a, s, ds);
}

fn dVecLength2(g: float4x4, v: dVec) -> Dual
{
    return g[0][0]*dSqr(v.a[0]) + g[1][1]*dSqr(v.a[1]) + g[2][2]*dSqr(v.a[2]) +
    custom.p1*( g[0][1]*dMul(v.a[0],v.a[1]) + g[0][2]*dMul(v.a[0],v.a[2]) +
           g[1][2]*dMul(v.a[1],v.a[2]) );
}

fn sqr(x: float) -> float
{
    return x*x;
}

fn diag(a: float4) -> float4x4
{
    return float4x4(
        a.x,0.0,0.0,0.0,
        0.0,a.y,0.0,0.0,
        0.0,0.0,a.z,0.0,
        0.0,0.0,0.0,a.w
    );
}


//optimized inverse of symmetric matrix
fn inverse_sym(m: float4x4) -> float4x4
{
	let n11 = m[0][0]; let n12 = m[1][0]; let n13 = m[2][0]; let n14 = m[3][0];
	let n22 = m[1][1]; let n23 = m[2][1]; let n24 = m[3][1];
	let n33 = m[2][2]; let n34 = m[3][2];
	let n44 = m[3][3];

	let t11 = 2.0 * n23 * n34 * n24 - n24 * n33 * n24 - n22 * n34 * n34 - n23 * n23 * n44 + n22 * n33 * n44;
	let t12 = n14 * n33 * n24 - n13 * n34 * n24 - n14 * n23 * n34 + n12 * n34 * n34 + n13 * n23 * n44 - n12 * n33 * n44;
	let t13 = n13 * n24 * n24 - n14 * n23 * n24 + n14 * n22 * n34 - n12 * n24 * n34 - n13 * n22 * n44 + n12 * n23 * n44;
	let t14 = n14 * n23 * n23 - n13 * n24 * n23 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

	let det = n11 * t11 + n12 * t12 + n13 * t13 + n14 * t14;
	let idet = 1.0 / det;

	var ret: float4x4;

	ret[0][0] = t11 * idet;
	ret[0][1] = (n24 * n33 * n14 - n23 * n34 * n14 - n24 * n13 * n34 + n12 * n34 * n34 + n23 * n13 * n44 - n12 * n33 * n44) * idet;
	ret[0][2] = (n22 * n34 * n14 - n24 * n23 * n14 + n24 * n13 * n24 - n12 * n34 * n24 - n22 * n13 * n44 + n12 * n23 * n44) * idet;
	ret[0][3] = (n23 * n23 * n14 - n22 * n33 * n14 - n23 * n13 * n24 + n12 * n33 * n24 + n22 * n13 * n34 - n12 * n23 * n34) * idet;

	ret[1][0] = ret[0][1];
	ret[1][1] = (2.0 * n13 * n34 * n14 - n14 * n33 * n14 - n11 * n34 * n34 - n13 * n13 * n44 + n11 * n33 * n44) * idet;
	ret[1][2] = (n14 * n23 * n14 - n12 * n34 * n14 - n14 * n13 * n24 + n11 * n34 * n24 + n12 * n13 * n44 - n11 * n23 * n44) * idet;
	ret[1][3] = (n12 * n33 * n14 - n13 * n23 * n14 + n13 * n13 * n24 - n11 * n33 * n24 - n12 * n13 * n34 + n11 * n23 * n34) * idet;

	ret[2][0] = ret[0][2];
	ret[2][1] = ret[1][2];
    ret[2][2] = (2.0 * n12 * n24 * n14 - n14 * n22 * n14 - n11 * n24 * n24 - n12 * n12 * n44 + n11 * n22 * n44) * idet;
	ret[2][3] = (n13 * n22 * n14 - n12 * n23 * n14 - n13 * n12 * n24 + n11 * n23 * n24 + n12 * n12 * n34 - n11 * n22 * n34) * idet;

	ret[3][0] = ret[0][3];
	ret[3][1] = ret[1][3];
	ret[3][2] = ret[2][3];
	ret[3][3] = (2.0 * n12 * n23 * n13 - n13 * n22 * n13 - n11 * n23 * n23 - n12 * n12 * n33 + n11 * n22 * n33) * idet;

	return ret;
}

fn KerrGetR2(p: float3) -> float
{
    let rho = dot(p,p) - sqr(custom.KerrA);
    let r2 = 0.5*(rho + sqrt(sqr(rho) + sqr(2.0*custom.KerrA*p.z)));
    return r2;
}

fn KerrGetK(p: float3) -> float4
{
    let r2 = KerrGetR2(p);
    let r = sqrt(r2);
    let invr2 = 1.0 / (r2 + sqr(custom.KerrA) + 1e-6);
    let  k = float3((r*p.x - custom.KerrA*p.y) * invr2, (r*p.y + custom.KerrA*p.x) * invr2, p.z/(r + 1e-6));
    let f = r2 * (2.0 * custom.KerrM * r - sqr(custom.KerrQ)) / (r2 * r2 + sqr(custom.KerrA * p.z) + 1e-6);
    return float4(k, f);
}

fn G(q: float4) -> float4x4
{
    //Kerr metric in Kerr-Schild coordinates
    let k = KerrGetK(q.yzw);
    let kf = k.w*float4(1.0, k.xyz);
    return diag(float4(-1.0,1.0,1.0,1.0)) + float4x4(kf, k.x*kf, k.y*kf, k.z*kf);
}

fn Ginv(q: float4) -> float4x4
{
    //inverse of Kerr metric in Kerr-Schild coordinates
    let k = KerrGetK(q.yzw);
    let kf = k.w*vec4(1.0, -k.xyz)/dot(k.xyz, k.xyz);
    return diag(float4(-1.0,1.0,1.0,1.0)) + float4x4(-kf, k.x*kf, k.y*kf, k.z*kf);
}

fn Lpotential(q: Dual) -> Dual
{
    return -custom.CosK*dSqr(q) +
    custom.Nonlinear*custom.CosK*dSqr(dSqr(clamp(q,float2(-0.5),float2(0.5))));
    //return custom.CosK*dCos(custom.CosA*q);
}

fn Pos(p: uint2) -> float4
{
    let pos = 0.0*float2(1.4, 0.0) + 45.5*(float2(p)/float(LENGTH) - 0.5);
    return float4(0.0,pos.x, pos.y,0.0);
}

#define TIMEK (8.0*custom.TimeK)

fn LoadA(a0: Dual, i: uint2, i0: uint2) -> Dual
{
    return select(dC(Load(i).x), a0, all(i==i0));
}

fn LoadA0(i: uint2) -> Dual
{
    let data = Load(i);
    return dC(data.x - data.y);
}

fn LoadA2(a0: Dual, i: uint2, i0: uint2) -> Dual
{
    return select(dC(Load(i).z), a0, all(i==i0));
}

fn LagrangianField(A2: Dual, A1: Dual, p: int2, dp: int2) -> Dual
{
    //load the variables
    let i = uint2(p + dp);
    let i0 = uint2(p);

    let a0 = LoadA0(i);
    let a1 = LoadA(A1, i, i0);
    let a2 = LoadA2(A2, i, i0);
    let ax0 = LoadA(A1, i - uint2(1u, 0u), i0);
    let ax2 = LoadA(A1, i + uint2(1u, 0u), i0);
    let ay0 = LoadA(A1, i - uint2(0u, 1u), i0);
    let ay2 = LoadA(A1, i + uint2(0u, 1u), i0);

    let a0x0 = LoadA0(i - uint2(1u, 0u));
    let a0x2 = LoadA0(i + uint2(1u, 0u));
    let a0y0 = LoadA0(i - uint2(0u, 1u));
    let a0y2 = LoadA0(i + uint2(0u, 1u));

    let a2x0 = LoadA2(A2, i - uint2(1u, 0u), i0);
    let a2x2 = LoadA2(A2, i + uint2(1u, 0u), i0);
    let a2y0 = LoadA2(A2, i - uint2(0u, 1u), i0);
    let a2y2 = LoadA2(A2, i + uint2(0u, 1u), i0);

    //compute the metric
    let x = Pos(i);
    let g = Ginv(x)*diag(float4(TIMEK,1.0,1.0,1.0));
    let vol = 0.5*sqrt(-1.0*determinant(g));

    let dadt0 = a2 - a1;
    let dadx0 = ax2 - a1;
    let dady0 = ay2 - a1;

    let dadt1 = a1 - a0;
    let dadx1 = a1 - ax0;
    let dady1 = a1 - ay0;

    let da0dx0 = a0x2 - a0;
    let da0dy0 = a0y2 - a0;
    let da0dx1 = a0 - a0x0;
    let da0dy1 = a0 - a0y0;

    let da2dx0 = a2x2 - a2;
    let da2dy0 = a2y2 - a2;
    let da2dx1 = a2 - a2x0;
    let da2dy1 = a2 - a2y0;

    let v0 = dVecMake(dadt0, dadx0, dady0);
    let v1 = dVecMake(dadt0, dadx0, dady1);
    let v2 = dVecMake(dadt0, dadx1, dady0);
    let v3 = dVecMake(dadt0, dadx1, dady1);
    let v4 = dVecMake(dadt1, dadx0, dady0);
    let v5 = dVecMake(dadt1, dadx0, dady1);
    let v6 = dVecMake(dadt1, dadx1, dady0);
    let v7 = dVecMake(dadt1, dadx1, dady1);

    let v0_1 = dVecMake(dadt1, da0dx0, da0dy0);
    let v1_1 = dVecMake(dadt1, da0dx0, da0dy1);
    let v2_1 = dVecMake(dadt1, da0dx1, da0dy0);
    let v3_1 = dVecMake(dadt1, da0dx1, da0dy1);
    let v4_1 = dVecMake(dadt0, da2dx0, da2dy0);
    let v5_1 = dVecMake(dadt0, da2dx0, da2dy1);
    let v6_1 = dVecMake(dadt0, da2dx1, da2dy0);
    let v7_1 = dVecMake(dadt0, da2dx1, da2dy1);

    return vol*(0.5*0.25*(
                dVecLength2(g, v0) + dVecLength2(g, v1) +
                dVecLength2(g, v2) + dVecLength2(g, v3) +
                dVecLength2(g, v4) + dVecLength2(g, v5) +
                dVecLength2(g, v6) + dVecLength2(g, v7)
                ) +
                0.5*0.25*(
                dVecLength2(g, v0_1) + dVecLength2(g, v1_1) +
                dVecLength2(g, v2_1) + dVecLength2(g, v3_1) +
                dVecLength2(g, v4_1) + dVecLength2(g, v5_1) +
                dVecLength2(g, v6_1) + dVecLength2(g, v7_1)
                ) - 12.0*Lpotential(a1));
}

//integral of the Lagrangian Field (here - just sum)
fn ActionIntegral(A2: Dual, A1: Dual, p: int2) -> Dual
{
    return LagrangianField(A2, A1, p, int2(0,0))  +
           LagrangianField(A2, A1, p, int2(1,0))  +
           LagrangianField(A2, A1, p, int2(0,1))  +
           LagrangianField(A2, A1, p, int2(-1,0)) +
           LagrangianField(A2, A1, p, int2(0,-1)) ;
}


#define NEWTON_STEPS 1
#define STEP_SIZE 1.0
fn Solve(i: uint2) -> float4
{
    let a = Load(i);
    let a0 = a.x - a.y;
    let a1 = a.x;
    var a2 = a.z; //predicted value

    var dA = 0.0;
    var dT = STEP_SIZE;
    var j = 0;
    for(; j<NEWTON_STEPS; j++)
    {
        let A0 = ActionIntegral(dC(a2), dV(a1), int2(i));
        let A1 = ActionIntegral(dC(a2 - 0.001), dV(a1), int2(i));
        let A2 = ActionIntegral(dC(a2 + 0.001), dV(a1), int2(i));

        dA = A0.y*0.002/(A2.y - A1.y);
        a2 -= dT*dA;

        if(abs(dA) < 1e-6)  { break; } //update step small enough
        //if(j == 64) { dT = 1.0; } //use full step for rapid convergence
    }

    if(SIM_FRAME%5u == 0u)
    {
        let x = Pos(i);
        let r = sqrt(KerrGetR2(x.yzw));
        let sdB = -sdBox(float2(i)/float(LENGTH) - 0.5, float2(0.5));
        let bk = smoothstep(0.0, 0.05, sdB);
        let k = smoothstep(1.0, 1.55, r)*mix(1.0,smoothstep(1.0, 1.6, r)*bk,0.035);

        let at = (a2 - a1)*k;
        a2 = k*(a1+at*custom.TimeStep);

        let aavg = mix(a.w, (a2*a2 + at*at), 0.3);

        return float4(a2,at,a2,aavg);
    }
    else
    {
        return float4(a.xy,a2,a.w);
    }
}

fn expi(x: float) -> float2
{
    return float2(cos(x), sin(x));
}

#dispatch_count Simulation ITERATIONS
#workgroup_count Simulation WG WG 1
@compute @workgroup_size(16,16)
fn Simulation(@builtin(global_invocation_id) id: uint3)
{
    if(id.x == 0u || id.x == uint(LENGTH) - 1u || id.y == 0u || id.y == uint(LENGTH) - 1u)
    {
        Store(id.xy, float4(0.0));
        return;
    }

    var f = Solve(id.xy);
    if(time.frame == 0u)
    {
        f *= 0.0;
    }
    if(time.frame == 0u)
    {
        let x = Pos(id.xy).yz;
        let a =0.0*exp(-dot(x,x));
        let dx = x - float2(16.0*custom.IDis, 0.0);
        let vel = float2(0.0,-32.0*custom.IVel);
        let b = 2.0*float2(1.0, 0.5) * exp(-8.0*custom.IRad*dot(dx, dx)) * expi(dot(dx, vel));
        f += float4(b,0.0,0.0);
    }

    if(time.frame == 60u && id.y > uint(LENGTH/2))
    {
        f*=0.0;
    }

    Store(id.xy, clamp(f, float4(-35.0), float4(35.0)));
}

fn Bilinear(p: float2) -> float4
{
    let pi = uint2(floor(p));
    let pf = fract(p);

    let a00 = Load(pi + uint2(0u,0u));
    let a01 = Load(pi + uint2(0u,1u));
    let a10 = Load(pi + uint2(1u,0u));
    let a11 = Load(pi + uint2(1u,1u));

    let a0 = mix(a00, a01, pf.y);
    let a1 = mix(a10, a11, pf.y);

    return mix(a0, a1, pf.x);
}

fn Sample(p0: float2) -> float4
{
    let screen_size = float2(textureDimensions(screen));
    var p = p0;
    if(mouse.click == 1)
    {
        p = 0.25*(p0 - screen_size*0.5) + float2(mouse.pos.xy);
    }
    let id = float(LENGTH) * ((p - screen_size.xy*0.5)/float(SCREEN_HEIGHT) + 0.5);
    let x = Pos(uint2(id));
    let g = G(x);
    let K = KerrGetK(x.yzw);
    let vol = sqrt(-1.0/determinant(g));

    let V = 0.5*float4(-0.4,float2(mouse.pos.xy)/screen_size - 0.5,0.);
    let V2 = 1.0*dot(g*V,V);

    return Bilinear(id) + 0.*float4(length(K));
}

@compute @workgroup_size(16, 16)
fn MainImage(@builtin(global_invocation_id) id: uint3)
{
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(id.y) + .5);

    var col = float3(2.00)*Sample(fragCoord).w;

    col = pow(col, float3(0.6));

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
