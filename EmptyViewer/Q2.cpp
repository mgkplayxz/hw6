#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define _CRT_SECURE_NO_WARNINGS
#define SCREEN_WIDTH 512
#define SCREEN_HEIGHT 512
#define MAX_VERTICES 10000

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    float x, y, z;
    float wx, wy, wz;
    float nx, ny, nz;
    unsigned char color[3]; 
} Vertex;

unsigned char framebuffer[SCREEN_HEIGHT][SCREEN_WIDTH][3];
float depthBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];
Vertex gVertexBuffer[MAX_VERTICES];
int gNumVertices = 0;

void clear_buffers() {
    for (int y = 0; y < SCREEN_HEIGHT; ++y) {
        for (int x = 0; x < SCREEN_WIDTH; ++x) {
            framebuffer[y][x][0] = 0;
            framebuffer[y][x][1] = 0;
            framebuffer[y][x][2] = 0;
            depthBuffer[y][x] = 1.0f;
        }
    }
}

void put_pixel(int x, int y, float z, unsigned char r, unsigned char g, unsigned char b) {

    y = SCREEN_HEIGHT - 1 - y;
    x = SCREEN_WIDTH - 1 - x;
    if (x < 0 || x >= SCREEN_WIDTH || y < 0 || y >= SCREEN_HEIGHT) return;
    if (z < depthBuffer[y][x]) {
        framebuffer[y][x][0] = r;
        framebuffer[y][x][1] = g;
        framebuffer[y][x][2] = b;
        depthBuffer[y][x] = z;
    }
}

void compute_vertex_color(float px, float py, float pz,
    float nx, float ny, float nz,
    unsigned char out_color[3]) {
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    nx /= len; ny /= len; nz /= len;

    float lx = -4 - px, ly = 4 - py, lz = -3 - pz;
    float lv_len = sqrtf(lx * lx + ly * ly + lz * lz);
    lx /= lv_len; ly /= lv_len; lz /= lv_len;

    float vx = -px, vy = -py, vz = -pz;
    float v_len = sqrtf(vx * vx + vy * vy + vz * vz);
    vx /= v_len; vy /= v_len; vz /= v_len;

    float hx = lx + vx, hy = ly + vy, hz = lz + vz;
    float h_len = sqrtf(hx * hx + hy * hy + hz * hz);
    hx /= h_len; hy /= h_len; hz /= h_len;

    float NdotL = fmaxf(0.0f, nx * lx + ny * ly + nz * lz);
    float NdotH = fmaxf(0.0f, nx * hx + ny * hy + nz * hz);

    float ka[3] = { 0.0f, 1.0f, 0.0f };
    float kd[3] = { 0.0f, 0.5f, 0.0f };
    float ks[3] = { 1.0f, 1.0f, 1.0f };
    float p = 16.0f;
    float Ia = 0.2f;

    float color[3];
    for (int i = 0; i < 3; ++i) {
        float ambient = ka[i] * Ia;
        float diffuse = kd[i] * NdotL;
        float specular = ks[i] * powf(NdotH, p);
        color[i] = ambient + diffuse + specular;
        color[i] = powf(fminf(color[i], 1.0f), 1.0f / 2.2f);
        out_color[i] = (unsigned char)(255.0f * color[i]);
    }
}


void rasterize_triangle(Vertex v0, Vertex v1, Vertex v2, unsigned char color[3]) {
    float ux = v1.wx - v0.wx;
    float uy = v1.wy - v0.wy;
    float uz = v1.wz - v0.wz;
    float vx = v2.wx - v0.wx;
    float vy = v2.wy - v0.wy;
    float vz = v2.wz - v0.wz;

    float nx = uy * vz - uz * vy;
    float ny = uz * vx - ux * vz;
    float nz = ux * vy - uy * vx;

    float cx = (v0.wx + v1.wx + v2.wx) / 3.0f;
    float cy = (v0.wy + v1.wy + v2.wy) / 3.0f;
    float cz = (v0.wz + v1.wz + v2.wz) / 3.0f;

    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    nx /= len; ny /= len; nz /= len;

    Vertex n0 = v0, n1 = v1, n2 = v2;
    n0.wx = nx; n0.wy = ny; n0.wz = nz;
    n1.wx = nx; n1.wy = ny; n1.wz = nz;
    n2.wx = nx; n2.wy = ny; n2.wz = nz;

    unsigned char color0[3], color1[3], color2[3];
    compute_vertex_color(v0.wx, v0.wy, v0.wz, v0.nx, v0.ny, v0.nz, color0);
    compute_vertex_color(v1.wx, v1.wy, v1.wz, v1.nx, v1.ny, v1.nz, color1);
    compute_vertex_color(v2.wx, v2.wy, v2.wz, v2.nx, v2.ny, v2.nz, color2);

    v0.color[0] = color0[0]; v0.color[1] = color0[1]; v0.color[2] = color0[2];
    v1.color[0] = color1[0]; v1.color[1] = color1[1]; v1.color[2] = color1[2];
    v2.color[0] = color2[0]; v2.color[1] = color2[1]; v2.color[2] = color2[2];

    float sx0 = v0.x, sy0 = v0.y, sz0 = v0.z;
    float sx1 = v1.x, sy1 = v1.y, sz1 = v1.z;
    float sx2 = v2.x, sy2 = v2.y, sz2 = v2.z;

    int minx = (int)fmaxf(0.0f, floorf(fminf(fminf(sx0, sx1), sx2)));
    int maxx = (int)fminf(SCREEN_WIDTH - 1, ceilf(fmaxf(fmaxf(sx0, sx1), sx2)));
    int miny = (int)fmaxf(0.0f, floorf(fminf(fminf(sy0, sy1), sy2)));
    int maxy = (int)fminf(SCREEN_HEIGHT - 1, ceilf(fmaxf(fmaxf(sy0, sy1), sy2)));

    float area = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0);
    if (fabsf(area) < 1e-5) return;

    for (int y = miny; y <= maxy; ++y) {
        for (int x = minx; x <= maxx; ++x) {
            float w0 = (sx1 - sx0) * (y - sy0) - (sy1 - sy0) * (x - sx0);
            float w1 = (sx2 - sx1) * (y - sy1) - (sy2 - sy1) * (x - sx1);
            float w2 = (sx0 - sx2) * (y - sy2) - (sy0 - sy2) * (x - sx2);
            if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0)) {
                float alpha = ((sx1 - x) * (sy2 - y) - (sx2 - x) * (sy1 - y)) / area;
                float beta = ((sx2 - x) * (sy0 - y) - (sx0 - x) * (sy2 - y)) / area;
                float gamma = 1.0f - alpha - beta;
                float z = alpha * sz0 + beta * sz1 + gamma * sz2;

                unsigned char r = (unsigned char)(alpha * v0.color[0] + beta * v1.color[0] + gamma * v2.color[0]);
                unsigned char g = (unsigned char)(alpha * v0.color[1] + beta * v1.color[1] + gamma * v2.color[1]);
                unsigned char b = (unsigned char)(alpha * v0.color[2] + beta * v1.color[2] + gamma * v2.color[2]);

                put_pixel(x, y, z, r, g, b);
            }
        }
    }
}



void create_scene() {
    int width = 32;
    int height = 16;
    float radius = 1.0f;
    for (int j = 1; j < height - 1; ++j) {
        for (int i = 0; i < width; ++i) {
            float theta = (float)i / (float)(width - 1) * 2.0f * M_PI;
            float phi = (float)j / (float)(height - 1) * M_PI;
            float x = sinf(phi) * cosf(theta);
            float y = cosf(phi);
            float z = sinf(phi) * sinf(theta);
            Vertex v;
            v.wx = x * radius;
            v.wy = y * radius;
            v.wz = z * radius - 3.0f;
            v.nx = x;
            v.ny = y;
            v.nz = z;
            gVertexBuffer[gNumVertices++] = v;
        }
    }

    Vertex top = { 0.0f, radius, -3.0f, 0.0f, 1.0f, 0.0f };
    Vertex bottom = { 0.0f, -radius, -3.0f, 0.0f, -1.0f, 0.0f };
    gVertexBuffer[gNumVertices++] = top;
    gVertexBuffer[gNumVertices++] = bottom;

}

void project_vertices() {
    float l = -0.1f, r = 0.1f, b = -0.1f, tproj = 0.1f, n = 0.1f, f = 1000.0f;
    float P[4][4] = { 0 };
    P[0][0] = 2.0f * n / (r - l);
    P[1][1] = 2.0f * n / (tproj - b);
    P[2][2] = -(f + n) / (f - n);
    P[2][3] = -(2.0f * f * n) / (f - n);
    P[3][2] = -1.0f;

    for (int i = 0; i < gNumVertices; ++i) {
        float x = gVertexBuffer[i].wx;
        float y = gVertexBuffer[i].wy;
        float z = gVertexBuffer[i].wz;

        float xp = P[0][0] * x;
        float yp = P[1][1] * y;
        float zp = P[2][2] * z + P[2][3];
        float wp = -z;

        xp /= wp; yp /= wp; zp /= wp;

        gVertexBuffer[i].x = (1.0f - xp) * 0.5f * SCREEN_WIDTH;
        gVertexBuffer[i].y = (yp + 1.0f) * 0.5f * SCREEN_HEIGHT;
        gVertexBuffer[i].z = (zp + 1.0f) * 0.5f;
    }
}


void render_scene() {
    int width = 32;
    int height = 16;
    int poleTop = (height - 2) * width;
    int poleBottom = poleTop + 1;
    unsigned char dummy_color[3] = { 255, 255, 255 };

    for (int y = 0; y < height - 3; ++y) {
        for (int x = 0; x < width; ++x) {
            int nextX = (x + 1) % width;
            int i0 = y * width + x;
            int i1 = y * width + nextX;
            int i2 = (y + 1) * width + x;
            int i3 = (y + 1) * width + nextX;
            rasterize_triangle(gVertexBuffer[i0], gVertexBuffer[i2], gVertexBuffer[i1], dummy_color);
            rasterize_triangle(gVertexBuffer[i1], gVertexBuffer[i2], gVertexBuffer[i3], dummy_color);
        }
    }

    for (int x = 0; x < width; ++x) {
        int nextX = (x + 1) % width;
        rasterize_triangle(gVertexBuffer[poleTop], gVertexBuffer[x], gVertexBuffer[nextX], dummy_color);
    }

    int base = (height - 3) * width;
    for (int x = 0; x < width; ++x) {
        int nextX = (x + 1) % width;
        rasterize_triangle(gVertexBuffer[poleBottom], gVertexBuffer[base + nextX], gVertexBuffer[base + x], dummy_color);
    }
}

void save_image(const char* filename) {
    FILE* f;
    fopen_s(&f, filename, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", SCREEN_WIDTH, SCREEN_HEIGHT);
    fwrite(framebuffer, 1, SCREEN_WIDTH * SCREEN_HEIGHT * 3, f);
    fclose(f);
}


int main() {
    clear_buffers();
    create_scene();
    project_vertices();
    render_scene();
    save_image("output.ppm");
    return 0;
}