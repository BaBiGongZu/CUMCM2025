#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

// --- 1. 全局常量和数据结构 ---
#define TARGET_CENTER_X 0.0
#define TARGET_CENTER_Y 200.0
#define TARGET_RADIUS 7.0
#define TARGET_HEIGHT 10.0
#define SMOKE_RADIUS 10.0

#define NUM_SIDE_POINTS 100
#define NUM_BOTTOM_POINTS 50
#define NUM_TOP_POINTS 50
#define TOTAL_KEY_POINTS (NUM_SIDE_POINTS + NUM_BOTTOM_POINTS + NUM_TOP_POINTS)
   
typedef struct {
    double x, y, z;
} Point3D;

Point3D g_key_points[TOTAL_KEY_POINTS];

// --- 2. 辅助函数 ---

// 初始化目标关键点
void initialize_key_points() {
    int index = 0;
    // 侧面点
    for (int i = 0; i < 10; ++i) { // 10个高度层
        double h = (double)i / 9.0 * TARGET_HEIGHT;
        for (int j = 0; j < 10; ++j) { // 每层10个点
            double angle = 2.0 * M_PI * j / 10.0;
            g_key_points[index++] = (Point3D){
                TARGET_CENTER_X + TARGET_RADIUS * cos(angle),
                TARGET_CENTER_Y + TARGET_RADIUS * sin(angle),
                h
            };
        }
    }
    // 底面点
    for (int i = 0; i < NUM_BOTTOM_POINTS; ++i) {
        double angle = 2.0 * M_PI * i / NUM_BOTTOM_POINTS;
        g_key_points[index++] = (Point3D){
            TARGET_CENTER_X + TARGET_RADIUS * cos(angle),
            TARGET_CENTER_Y + TARGET_RADIUS * sin(angle),
            0.0
        };
    }
    // 顶面点
    for (int i = 0; i < NUM_TOP_POINTS; ++i) {
        double angle = 2.0 * M_PI * i / NUM_TOP_POINTS;
        g_key_points[index++] = (Point3D){
            TARGET_CENTER_X + TARGET_RADIUS * cos(angle),
            TARGET_CENTER_Y + TARGET_RADIUS * sin(angle),
            TARGET_HEIGHT
        };
    }
}

// 点到线段的距离
double point_to_line_distance(Point3D p, Point3D a, Point3D b) {
    Point3D ab = {b.x - a.x, b.y - a.y, b.z - a.z};
    Point3D ap = {p.x - a.x, p.y - a.y, p.z - a.z};
    double line_length_sq = ab.x * ab.x + ab.y * ab.y + ab.z * ab.z;
    if (line_length_sq < 1e-12) {
        return sqrt(ap.x * ap.x + ap.y * ap.y + ap.z * ap.z);
    }
    double dot_product = ap.x * ab.x + ap.y * ab.y + ap.z * ab.z;
    double projection = dot_product / line_length_sq;
    if (projection < 0) {
        return sqrt(ap.x * ap.x + ap.y * ap.y + ap.z * ap.z);
    }
    if (projection > 1) {
        Point3D bp = {p.x - b.x, p.y - b.y, p.z - b.z};
        return sqrt(bp.x * bp.x + bp.y * bp.y + bp.z * bp.z);
    }
    Point3D closest_point = {a.x + projection * ab.x, a.y + projection * ab.y, a.z + projection * ab.z};
    Point3D dist_vec = {p.x - closest_point.x, p.y - closest_point.y, p.z - closest_point.z};
    return sqrt(dist_vec.x * dist_vec.x + dist_vec.y * dist_vec.y + dist_vec.z * dist_vec.z);
}

// 检查多个烟雾云是否完全遮蔽视线
bool check_multiple_clouds_blocking_internal(Point3D missile_pos, const Point3D* cloud_positions, int num_clouds) {
    if (num_clouds == 0) return false;

    for (int i = 0; i < TOTAL_KEY_POINTS; ++i) {
        Point3D key_point = g_key_points[i];
        bool kp_is_blocked = false;
        for (int j = 0; j < num_clouds; ++j) {
            if (point_to_line_distance(cloud_positions[j], missile_pos, key_point) <= SMOKE_RADIUS) {
                kp_is_blocked = true;
                break;
            }
        }
        if (!kp_is_blocked) {
            return false; // 只要有一个关键点没被挡住，就返回false
        }
    }
    return true; // 所有关键点都被挡住了
}

// --- 3. 导出的 DLL 函数 ---

DLL_EXPORT const char* get_version() {
    return "2.0.0";
}

DLL_EXPORT int get_key_points_count() {
    return TOTAL_KEY_POINTS;
}

DLL_EXPORT void get_sampling_info(int* total, int* bottom, int* top, int* side) {
    *total = TOTAL_KEY_POINTS;
    *bottom = NUM_BOTTOM_POINTS;
    *top = NUM_TOP_POINTS;
    *side = NUM_SIDE_POINTS;
}

// 新的核心函数：一次性计算总遮蔽时长
DLL_EXPORT double calculate_total_blocking_duration(
    const double* missile_start_arr,
    const double* missile_velocity_arr,
    const double* explode_positions_flat,
    const double* explode_times_arr,
    int num_clouds,
    double total_flight_time,
    double time_step,
    double sink_speed,
    double smoke_duration
) {
    // 确保关键点已初始化
    static bool initialized = false;
    if (!initialized) {
        initialize_key_points();
        initialized = true;
    }

    Point3D missile_start = {missile_start_arr[0], missile_start_arr[1], missile_start_arr[2]};
    Point3D missile_velocity = {missile_velocity_arr[0], missile_velocity_arr[1], missile_velocity_arr[2]};

    int blocked_steps = 0;
    int num_time_steps = (int)(total_flight_time / time_step);

    Point3D active_cloud_positions[10]; // 假设最多10个烟雾弹

    for (int i = 0; i < num_time_steps; ++i) {
        double t = i * time_step;
        
        Point3D current_missile_pos = {
            missile_start.x + missile_velocity.x * t,
            missile_start.y + missile_velocity.y * t,
            missile_start.z + missile_velocity.z * t
        };

        int num_active_clouds = 0;
        for (int j = 0; j < num_clouds; ++j) {
            double t_explode = explode_times_arr[j];
            if (t >= t_explode && t < (t_explode + smoke_duration)) {
                double time_since_explode = t - t_explode;
                
                active_cloud_positions[num_active_clouds].x = explode_positions_flat[j * 3 + 0];
                active_cloud_positions[num_active_clouds].y = explode_positions_flat[j * 3 + 1];
                active_cloud_positions[num_active_clouds].z = explode_positions_flat[j * 3 + 2] - sink_speed * time_since_explode;
                
                // 确保烟雾云在地面以上
                if (active_cloud_positions[num_active_clouds].z > 0) {
                    num_active_clouds++;
                }
            }
        }

        if (num_active_clouds > 0) {
            if (check_multiple_clouds_blocking_internal(current_missile_pos, active_cloud_positions, num_active_clouds)) {
                blocked_steps++;
            }
        }
    }

    return blocked_steps * time_step;
}