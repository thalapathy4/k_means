#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <raylib.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <raymath.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024
#define K 4 
#define SAMPLE_RADIUS 4.0f
#define MEAN_RADIUS (2*SAMPLE_RADIUS)

float euclidean_distance(Vector2 a, Vector2 b);
float silhouette_score(void);

typedef struct {
    Vector2 *items;
    size_t count;
    size_t capacity;
} Samples;

static Samples set = {0};
static Samples cluster[K] = {0};
static Vector2 means[K] = {0};
static Color raylib_colors[] = {YELLOW, PINK, WHITE, RED, BLUE, BROWN, GREEN, ORANGE, PURPLE, SKYBLUE};
static size_t colors_count = sizeof(raylib_colors) / sizeof(raylib_colors[0]);

static float min_x = FLT_MAX, max_x = -FLT_MAX, min_y = FLT_MAX, max_y = -FLT_MAX;

static inline float rand_float(void) {
    return (float)rand() / RAND_MAX;
}

static inline float lerpf(float random, float min, float max) {
    return (random * (max - min) + min);
}

void read_csv_file(const char* filename, Samples* samples) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Failed to open file: %s\n", filename);
        return;
    }

    char line[MAX_LINE_LENGTH];
    samples->count = 0;
    samples->capacity = 601;  
    samples->items = (Vector2*)malloc(samples->capacity * sizeof(Vector2));

    // Skip the header line
    fgets(line, MAX_LINE_LENGTH, file);

    while (fgets(line, MAX_LINE_LENGTH, file)) {
        int id;
        char gender[10];
        int age;
        float income, score;
        if (sscanf(line, "%d,%[^,],%d,%f,%f", &id, gender, &age, &income, &score) == 5) {
            samples->items[samples->count++] = (Vector2){income, score};

            // Update min and max value
            if (income < min_x) min_x = income;
            if (income > max_x) max_x = income;
            if (score < min_y) min_y = score;
            if (score > max_y) max_y = score;
        }
    }

    fclose(file);
}

Vector2 project_sample_to_screen(Vector2 sample) {
    float x = (sample.x - min_x) / (max_x - min_x);
    float y = (sample.y - min_y) / (max_y - min_y);
    return CLITERAL(Vector2) { .x = GetScreenWidth()*x, .y = GetScreenHeight() - GetScreenHeight()*y};
}

void generate_state(void) {
    set.count = 0;
    read_csv_file("customer_segmentation_expanded600.csv", &set);

    for(size_t i = 0; i < K; ++i) {
        means[i].x = lerpf(rand_float(), min_x, max_x);
        means[i].y = lerpf(rand_float(), min_y, max_y);
    }
}

void recluster_state(void) {
    for(size_t i = 0; i < K; ++i) {
        cluster[i].count = 0;
        free(cluster[i].items);
        cluster[i].items = NULL;
    }

    for(size_t i = 0; i < set.count; ++i) {
        Vector2 p = set.items[i];
        int k = -1;
        float s = FLT_MAX;
        for(size_t j = 0; j < K; ++j) {
            Vector2 m = means[j];
            float sm = Vector2LengthSqr(Vector2Subtract(p, m));
            if(sm < s) {
                k = j;
                s = sm;
            }
        }
        cluster[k].count += 1;
        cluster[k].items = (Vector2 *)realloc(cluster[k].items, sizeof(Vector2) * cluster[k].count);
        cluster[k].items[cluster[k].count - 1] = p;
    }
}

void update_means(void) {
    for(size_t i = 0; i < K; ++i) {
        if(cluster[i].count > 0) {
            means[i] = Vector2Zero();
            for(size_t j = 0; j < cluster[i].count; ++j) {
                means[i] = Vector2Add(means[i], cluster[i].items[j]);
            }
            means[i].x /= cluster[i].count;
            means[i].y /= cluster[i].count;
        } else {
            means[i].x = lerpf(rand_float(), min_x, max_x);
            means[i].y = lerpf(rand_float(), min_y, max_y);
        }
    }
}
float euclidean_distance(Vector2 a, Vector2 b) {
    return sqrtf(powf(b.x - a.x, 2) + powf(b.y - a.y, 2));
}

float silhouette_score(void) {
    float total_score = 0.0f;

    for (size_t i = 0; i < set.count; ++i) {
        Vector2 p = set.items[i];
        int current_cluster = -1;
        float a = 0.0f;
        float b = FLT_MAX;

        // Find the cluster of the current point
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < cluster[k].count; ++j) {
                if (Vector2Equals(p, cluster[k].items[j])) {
                    current_cluster = k;
                    break;
                }
            }
            if (current_cluster != -1) break;
        }

        if (current_cluster == -1) continue;

        // Calculate average distance to points in the same cluster
        float intra_cluster_distance = 0.0f;
        int same_cluster_count = 0;
        for (size_t j = 0; j < cluster[current_cluster].count; ++j) {
            if (!Vector2Equals(p, cluster[current_cluster].items[j])) {
                intra_cluster_distance += euclidean_distance(p, cluster[current_cluster].items[j]);
                ++same_cluster_count;
            }
        }
        a = (same_cluster_count > 0) ? intra_cluster_distance / same_cluster_count : 0.0f;

        // Calculate average distance to points in the nearest cluster
        for (size_t k = 0; k < K; ++k) {
            if (k == current_cluster) continue;
            float inter_cluster_distance = 0.0f;
            int other_cluster_count = 0;
            for (size_t j = 0; j < cluster[k].count; ++j) {
                inter_cluster_distance += euclidean_distance(p, cluster[k].items[j]);
                ++other_cluster_count;
            }
            float average_inter_cluster_distance = (other_cluster_count > 0) ? inter_cluster_distance / other_cluster_count : FLT_MAX;
            b = fminf(b, average_inter_cluster_distance);
        }

        // Compute silhouette score for this point
        float s = (b - a) / fmaxf(a, b);
        total_score += s;
    }

    // Return the average silhouette score
    return set.count > 0 ? total_score / set.count : 0.0f;
}



int main(void) {
    srand(time(0));
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(800, 600, "Customer Segmentation K-means");
    
    generate_state();
    if (set.count == 0) {
        printf("No data points loaded. Exiting.\n");
        CloseWindow();
        return 1;
    }
    recluster_state();

    while(!WindowShouldClose()) {
        if(IsKeyPressed(KEY_R)) {
            generate_state();
            recluster_state();
        }

        assert(K <= colors_count);
        
        if(IsKeyPressed(KEY_SPACE)) {
            update_means();
            recluster_state();
            float score = silhouette_score();
            printf("Silhouette Score: %.4f\n", score);  // Print silhouette score to console
        }
        
        BeginDrawing();
        ClearBackground(GetColor(0x181818AA));

        for(size_t i = 0; i < K; ++i) {
            Color color = raylib_colors[i % colors_count];
            Vector2 it = means[i];
            DrawCircleV(project_sample_to_screen(it), MEAN_RADIUS, color);
            
            for(size_t j = 0; j < cluster[i].count; ++j) {
                Vector2 it = cluster[i].items[j];
               DrawCircleV(project_sample_to_screen(it), SAMPLE_RADIUS, color);
            }
        }
  

          // Add X-axis label
          DrawText("Annual Income", GetScreenWidth() / 2 - 50, GetScreenHeight() - 20, 20, RAYWHITE);

          // Add Y-axis label (vertical)
          DrawTextPro(GetFontDefault(), "Spending Score", (Vector2){ 20, GetScreenHeight() / 2 }, (Vector2){ 0, 0 }, 90.0f, 20, 1, RAYWHITE);


    
        // DrawText("X: Annual Income", GetScreenWidth() - 200, GetScreenHeight() - 60, 20, RAYWHITE);
        // DrawText("Y: Spending Score", GetScreenWidth() - 200, GetScreenHeight() - 30, 20, RAYWHITE);

        EndDrawing();
    }

    for(size_t i = 0; i < K; ++i) {
        free(cluster[i].items);
    }
    free(set.items);
    CloseWindow();
    return 0;
}