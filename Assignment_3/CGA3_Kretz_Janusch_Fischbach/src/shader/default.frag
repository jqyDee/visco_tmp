#version 330 core

uniform bool checkerboard;

in vec4 tColor;
in vec3 tFragPos;
out vec4 FragColor;

void main(void)
{
    if (!checkerboard)
    {
        FragColor = tColor;
    } 
    else
    {
        vec3 color1 = vec3(0.0f); 
        vec3 color2 = vec3(0.5f); 
        vec3 texColor = mix(color1, color2, 0.5 * mod(floor(tFragPos.x) + floor(tFragPos.y) + floor(tFragPos.z), 2));
        FragColor = vec4(texColor + vec3(tColor) * 0.5, tColor.z);
    }
}
