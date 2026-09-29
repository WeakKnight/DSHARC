import common;

// An analytic art-directable sky, not a physical atmosphere LUT model.
// Sun intensity is irradiance on a surface perpendicular to the disk axis.
struct Lighting
{
    float3 sun_direction;
    float3 sun_color;
    float sun_intensity;
    float sun_cos_radius;
    float sky_intensity;
    float3 sky_zenith;
    float3 sky_horizon;
    float3 sky_ground;

    float3 sky(float3 direction)
    {
        if (direction.y >= 0.0)
            return sky_intensity * lerp(sky_horizon, sky_zenith, pow(direction.y, 0.45));
        return sky_intensity * lerp(sky_horizon, sky_ground, sqrt(-direction.y));
    }

    float sun_pdf()
    {
        return 1.0 / (2.0 * PI * (1.0 - sun_cos_radius));
    }

    float3 sun_radiance()
    {
        // Integral of cos(theta) over a cone is PI * sin(radius)^2.
        return sun_color * sun_intensity / (PI * (1.0 - sun_cos_radius * sun_cos_radius));
    }

    bool in_sun(float3 direction)
    {
        return sun_intensity > 0.0 && dot(direction, sun_direction) >= sun_cos_radius;
    }

    float3 sample_sun(float2 u)
    {
        float z = lerp(1.0, sun_cos_radius, u.x);
        float radius = sqrt(max(0.0, 1.0 - z * z));
        float phi = 2.0 * PI * u.y;
        return normalize(local_to_world(float3(radius * cos(phi), radius * sin(phi), z), sun_direction));
    }
};
