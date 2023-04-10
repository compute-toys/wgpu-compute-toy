//=== original link for citation: https://www.shadertoy.com/view/llySRh
//find many other tricks here: https://shadertoyunofficial.wordpress.com/

// --- short approx hue -------------- https://www.shadertoy.com/view/ll2cDc

fn hue(v: float) -> float4 {
    return .6 + .6 * cos( 6.3*(v)  + float4(0.,23.,21.,0.)  );
}


// --- printing chars, integers and floats ---------------------------

// --- access to the image of ascii code c

fn char_(p: float2, c: int) -> float4 {
    if (p.x<.0|| p.x>1. || p.y<0.|| p.y>1.) {
        return float4(0.,0.,0.,1e5);
    }
	return textureSampleLevel( channel0, trilinear,
        float2(p.x,1.-p.y)/16. + fract( float2(float(c), float(c/16)) / 16. ), 0.);
}

// --- display int4

fn pInt(p: float2, _n: float) -> float4 {
    var n = _n;
    var v = float4();
    if (n < 0.) {
        v += char_(p - float2(-.5,0.), 45 );
        n = -n;
    }

    for (var i = 3.; i>=0.; i -= 1.) {
        n /=  9.999999; // 10., // for windows :-(
        v += char_(p - .5*float2(i,0.), 48+ int(fract(n)*10.) );
    }
    return v;
}

// --- display float4.4

fn pFloat(_p: float2, _n: float) -> float4 {
    var p = _p;
    var n = _n;
    var v = float4();
    if (n < 0.) { v += char_(p - float2(-.5,0.), 45 ); n = -n; }
    var upper = floor(n);
    var lower = fract(n)*1e4 + .5;  // mla fix for rounding lost decimals
    if (lower >= 1e4) { lower -= 1e4; upper += 1.; }
    v += pInt(p,upper); p.x -= 2.;
    v += char_(p, 46);   p.x -= .5;
    v += pInt(p,lower);
    return v;
}

// printing full IEEE floats (right or left justified): see https://www.shadertoy.com/view/7dfyRH , https://www.shadertoy.com/view/7sscz7

// --- chars

var<private> O: float4;
var<private> U: float2;
var<private> CAPS: int;

fn C(c: int) {
    U.x -= .5;
    O+= char_(U, 64 + 32 * (1-CAPS) + c);
}

// NB: use either char.x ( pixel mask ) or char.w ( distance field + 0.5 )

// --- antialiased line drawing ------ https://www.shadertoy.com/view/4dcfW8

fn S(d: float, r: float, pix: float) -> float {
    return smoothstep( .75, -.75, (d)/(pix)-(r));   // antialiased draw. r >= 1.
}
// segment with disc ends: seamless distance to segment
fn line_(_p: float2, a: float2, _b: float2) -> float {
    let p = _p - a;
    let b = _b - a;
    let h = clamp(dot(p, b) / dot(b, b), 0., 1.);   // proj coord on line
    return length(p - b * h);                         // dist to segment
}
// line segment without disc ends ( sometime useful with semi-transparency )
fn line0(_p: float2, a: float2, _b: float2) -> float {
    let p = _p - a;
    let b = _b - a;
    let h = dot(p, b) / dot(b, b);                  // proj coord on line
    let c = clamp(h, 0., 1.);
    return select(1e5, length(p - b * h), h==c);            // dist to strict segment
}
    // You might directly return smoothstep( 3./R.y, 0., dist),
    //     but more efficient to factor all lines.
    // Indeed we can even return dot(,) and take sqrt at the end of polyline:
    // p -= b*h; return dot(p,p);


// for polylines with acute angles, see: https://www.shadertoy.com/view/fdVXRh


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / float2(screen_size.yy);

    // Time varying pixel colour
    var col = .5 + .5 * cos(time.elapsed + uv.xyx + float3(0.,2.,4.));

    let R = float2(screen_size);

    let lod = int(time.elapsed % 10.);

    U = ( uv - float2(.0,.9) ) * 16.;  CAPS=1; C(18); CAPS=0; C(5);C(19);C(15);C(12); CAPS=1; C(-6);  // "Resol"
                             U.x-=1.;  CAPS=0; C(19);C(3);C(18);C(5);C(5);C(14);               // "screen"
    U = ( uv - float2(.6,.9) ) * 16.;  CAPS=0; C(20);C(5);C(24);C(20);                       // "text"
    U = ( uv - float2(.85,.9) ) * 16.;  CAPS=0; C(12);C(15);C(4); U.x-=.5; C(-48+lod);             // "lod"

    U = ( uv - float2(.0,.6) ) * 16.;  CAPS=1; C(13); CAPS=0; C(15);C(21);C(19);C(5); CAPS=1; C(-6);  // "mouse"
    U = ( uv - float2(.5,.6) ) * 16.;  CAPS=1; C(20); CAPS=0; C(9);C(13);C(5); CAPS=1; C(-6);        // "Time"
    U = ( uv - float2(1.45,.55) ) * 16.;  CAPS=1; C(11); CAPS=0; C(5);C(25); CAPS=1; C(-6);         // "Key"

    U = ( uv - float2(.1,.8) ) * 8.;        // --- column 1
    O += pInt(U, R.x);  U.y += .8;   // window resolution
    O += pInt(U, R.y);  U.y += .8;
    O += pFloat((U-float2(-1.,.35))*1.5, R.x/R.y);  U.y += .8;
    U.y += .8;

    O += pInt(U, float(mouse.pos.x));  U.y += .8;        // mouse location
    O += pInt(U, float(mouse.pos.y));  U.y += .8;
    U.y += .4;
    O += pInt(U, float(mouse.click));  U.y += .8;

    U = ( uv - float2(.5,.8) ) * 8.;        // --- column 2

    //if ( !(textureDimensions(channel1).x > 0) ) {                  // texture not loaded yet
    //    if (U.x>0. && U.y>-1.5 && U.x<2.5 && U.y<1.5) { O.r+= .5; }
    //}
    O += pInt(U, float(textureDimensions(channel1).x));  U.y += .8; // texture ( video )
    O += pInt(U, float(textureDimensions(channel1).y));  U.y += .8; // see LOD in column 2b
    U.y += .8;

    O += pFloat(U, time.elapsed);         U.y += .8;  // time
    O += pInt(U, float(time.frame));   U.y += .8;  // iFrame

    U = ( uv - float2(.8,.8) ) * 8.;        // --- column 2b

    let Sz = textureDimensions(channel1,lod);
    O += pInt(U, float(Sz.x));  U.y += .8; // texture LOD
    O += pInt(U, float(Sz.y));  U.y += .4;

    U = ( uv - float2(1.4,.45) ) * 8.;       // --- column 4

    var b = false;
    for (var i=0; i<256; i++) {
        if (keyDown(uint(i)) )  {
            O += pInt(U, float(i));  // keypressed ascii
            b=true; U.y += .1 *8.;
        }
    }
    if (!b) { O += pInt(U, -1.); }

    O = O.xxxx;

    // --- non-fonts stuff

    U = (uv*R.y/R- .9)/.1;
    if (min(U.x,U.y)>0.) {
        O = hue(U.x);  // --- hue (already in sRGB final space)
    }

    U = (uv -float2(.9*R.x/R.y,.8))*10.;              // --- line drawing
    let pix = 10./R.y;               // pixel size
    O+= S( line_( U,float2(0.,0.),float2(1.1,.85)), 3., pix);
    O+= S( line0(U,float2(0.5,0.),float2(1.6,.85)), 3., pix);

    U = (uv - .8*R/R.y)*10.;                        // --- circle, discs, transp and blend
    O += S( abs(length(U-float2(.2,1.)) - .5), 1., pix); // disc. -.5: relative units. 1: pixel units
    O += S( length(U-float2(1.1,1.)) - .5, 0., pix) * float4(1.,0.,0.,1.)*.5; // float4(pureCol)*opacity
    O += (1.-O.a)*S( length(U-float2(1.1,.3)) - .5, 0., pix) * float4(0.,1.,0.,1.); // blend below prevs
    let C = S( length(U-float2(1.1,-.3)) - .5, 0., pix) * float4(0.,0.,1.,1.)*.5;  // blend above prevs
    O = C + (1.-C.a)*O;

    col = O.rgb;

    // Convert from gamma-encoded to linear colour space
    col = pow(col, float3(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
