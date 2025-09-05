#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

// 平台特定定义
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

// 全局常量（与Python程序保持一致）
static const int NUM_KEY_POINTS = 100;  // 改进的采样方式：30个底面 + 30个顶面 + 40个侧面
static const int NUM_BOTTOM_POINTS = 30;   // 底面圆周点数
static const int NUM_TOP_POINTS = 30;      // 顶面圆周点数
static const int NUM_SIDE_POINTS = 40;     // 侧面点数
static const double REAL_TARGET_CENTER[3] = {0.0, 200.0, 0.0};
static const double REAL_TARGET_RADIUS = 7.0;
static const double REAL_TARGET_HEIGHT = 10.0;
static const double R_SMOKE = 10.0;
static const double M_PI_LOCAL = 3.14159265358979323846;

// 计算点到线段的距离
EXPORT double point_to_line_distance(const double point[3], 
                                    const double line_start[3], 
                                    const double line_end[3]) {
    double line_vec[3] = {
        line_end[0] - line_start[0],
        line_end[1] - line_start[1],
        line_end[2] - line_start[2]
    };
    
    double line_length_sq = line_vec[0]*line_vec[0] + 
                           line_vec[1]*line_vec[1] + 
                           line_vec[2]*line_vec[2];
    
    if (line_length_sq < 1e-10) {
        double dist_vec[3] = {
            point[0] - line_start[0],
            point[1] - line_start[1],
            point[2] - line_start[2]
        };
        return sqrt(dist_vec[0]*dist_vec[0] + 
                   dist_vec[1]*dist_vec[1] + 
                   dist_vec[2]*dist_vec[2]);
    }
    
    double point_vec[3] = {
        point[0] - line_start[0],
        point[1] - line_start[1],
        point[2] - line_start[2]
    };
    
    double projection = (point_vec[0]*line_vec[0] + 
                        point_vec[1]*line_vec[1] + 
                        point_vec[2]*line_vec[2]) / line_length_sq;
    
    double closest_point[3];
    if (projection < 0.0) {
        closest_point[0] = line_start[0];
        closest_point[1] = line_start[1];
        closest_point[2] = line_start[2];
    } else if (projection > 1.0) {
        closest_point[0] = line_end[0];
        closest_point[1] = line_end[1];
        closest_point[2] = line_end[2];
    } else {
        closest_point[0] = line_start[0] + projection * line_vec[0];
        closest_point[1] = line_start[1] + projection * line_vec[1];
        closest_point[2] = line_start[2] + projection * line_vec[2];
    }
    
    double dist_vec[3] = {
        point[0] - closest_point[0],
        point[1] - closest_point[1],
        point[2] - closest_point[2]
    };
    
    return sqrt(dist_vec[0]*dist_vec[0] + 
               dist_vec[1]*dist_vec[1] + 
               dist_vec[2]*dist_vec[2]);
}

// 生成目标关键点 - 改进的采样方式
EXPORT void generate_target_key_points(double key_points[][3]) {
    int idx = 0;
    
    // 1. 底面圆周采样（z=0）
    for (int i = 0; i < NUM_BOTTOM_POINTS; i++) {
        double angle = 2.0 * M_PI_LOCAL * i / NUM_BOTTOM_POINTS;
        key_points[idx][0] = REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * cos(angle);
        key_points[idx][1] = REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * sin(angle);
        key_points[idx][2] = 0.0;
        idx++;
    }
    
    // 2. 顶面圆周采样（z=10）
    for (int i = 0; i < NUM_TOP_POINTS; i++) {
        double angle = 2.0 * M_PI_LOCAL * i / NUM_TOP_POINTS;
        key_points[idx][0] = REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * cos(angle);
        key_points[idx][1] = REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * sin(angle);
        key_points[idx][2] = REAL_TARGET_HEIGHT;
        idx++;
    }
    
    // 3. 圆柱侧面采样 - 在不同高度和角度均匀分布
    int points_per_height = 8;  // 每个高度层8个点
    int height_layers = NUM_SIDE_POINTS / points_per_height;  // 5个高度层
    
    for (int h = 0; h < height_layers; h++) {
        double z = REAL_TARGET_HEIGHT * (h + 1) / (height_layers + 1);  // 均匀分布在高度上
        for (int i = 0; i < points_per_height; i++) {
            double angle = 2.0 * M_PI_LOCAL * i / points_per_height;
            key_points[idx][0] = REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * cos(angle);
            key_points[idx][1] = REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * sin(angle);
            key_points[idx][2] = z;
            idx++;
        }
    }
}

// 检查单个烟雾云是否遮蔽某个关键点
bool is_point_blocked_by_cloud(const double key_point[3], 
                              const double missile_pos[3], 
                              const double cloud_pos[3]) {
    double dist = point_to_line_distance(cloud_pos, missile_pos, key_point);
    return dist <= R_SMOKE;
}

// 检查多个烟雾弹是否完全遮蔽目标（新增功能）
EXPORT bool check_multiple_clouds_blocking(const double missile_pos[3], 
                                         const double cloud_positions[][3], 
                                         int num_clouds) {
    static double key_points[100][3];
    static bool initialized = false;
    
    // 初始化关键点（只做一次）
    if (!initialized) {
        generate_target_key_points(key_points);
        initialized = true;
    }
    
    // 检查每个关键点是否至少被一个烟雾弹遮蔽
    for (int i = 0; i < NUM_KEY_POINTS; i++) {
        bool point_blocked = false;
        
        // 检查是否被任意一个烟雾弹遮蔽
        for (int j = 0; j < num_clouds; j++) {
            if (is_point_blocked_by_cloud(key_points[i], missile_pos, cloud_positions[j])) {
                point_blocked = true;
                break;  // 只要有一个烟雾弹遮蔽该点就足够了
            }
        }
        
        if (!point_blocked) {
            return false;  // 有关键点未被任何烟雾弹遮蔽
        }
    }
    
    return true;  // 所有关键点都被至少一个烟雾弹遮蔽
}

// 主要的遮蔽检测函数（保持向后兼容）
EXPORT bool check_complete_blocking(const double missile_pos[3], const double cloud_pos[3]) {
    // 使用新的多烟雾弹函数，传入单个烟雾弹
    double single_cloud[1][3];
    single_cloud[0][0] = cloud_pos[0];
    single_cloud[0][1] = cloud_pos[1];
    single_cloud[0][2] = cloud_pos[2];
    
    return check_multiple_clouds_blocking(missile_pos, single_cloud, 1);
}

// 批量计算多烟雾弹遮蔽时长的高效函数（新增功能）
EXPORT double calculate_multiple_clouds_blocking_duration(const double missile_start[3],
                                                        const double missile_velocity[3],
                                                        const double explode_positions[][3],
                                                        int num_clouds,
                                                        double t_start,
                                                        double t_end,
                                                        double time_step,
                                                        double sink_speed) {
    double total_blocked_time = 0.0;
    int num_steps = (int)((t_end - t_start) / time_step) + 1;
    
    for (int i = 0; i < num_steps; i++) {
        double t = t_start + i * time_step;
        if (t > t_end) break;
        
        // 当前导弹位置
        double current_missile_pos[3] = {
            missile_start[0] + t * missile_velocity[0],
            missile_start[1] + t * missile_velocity[1],
            missile_start[2] + t * missile_velocity[2]
        };
        
        // 当前所有烟雾云位置（考虑下沉）
        double time_since_explode = t - t_start;
        double current_cloud_positions[num_clouds][3];
        bool any_cloud_active = false;
        
        for (int j = 0; j < num_clouds; j++) {
            current_cloud_positions[j][0] = explode_positions[j][0];
            current_cloud_positions[j][1] = explode_positions[j][1];
            current_cloud_positions[j][2] = explode_positions[j][2] - sink_speed * time_since_explode;
            
            // 检查是否有烟雾云还在空中
            if (current_cloud_positions[j][2] > 0.0) {
                any_cloud_active = true;
            }
        }
        
        // 如果所有云团都落地，停止计算
        if (!any_cloud_active) {
            break;
        }
        
        // 检查是否完全遮蔽
        if (check_multiple_clouds_blocking(current_missile_pos, current_cloud_positions, num_clouds)) {
            total_blocked_time += time_step;
        }
    }
    
    return total_blocked_time;
}

// 批量计算遮蔽时长的高效函数（保持向后兼容）
EXPORT double calculate_blocking_duration_batch(const double missile_start[3],
                                              const double missile_velocity[3],
                                              const double explode_pos[3],
                                              double t_start,
                                              double t_end,
                                              double time_step,
                                              double sink_speed) {
    // 使用新的多烟雾弹函数，传入单个烟雾弹
    double single_explode_pos[1][3];
    single_explode_pos[0][0] = explode_pos[0];
    single_explode_pos[0][1] = explode_pos[1];
    single_explode_pos[0][2] = explode_pos[2];
    
    return calculate_multiple_clouds_blocking_duration(missile_start, missile_velocity, 
                                                     single_explode_pos, 1, 
                                                     t_start, t_end, time_step, sink_speed);
}

// 获取版本信息
EXPORT const char* get_version() {
    return "1.1.0";  // 更新版本号，表示增加了多烟雾弹功能
}

// 获取关键点数量 - 添加缺失的函数
EXPORT int get_key_points_count() {
    return NUM_KEY_POINTS;
}

// 获取第一个关键点的坐标 - 添加缺失的函数
EXPORT void get_first_key_point(double* x, double* y, double* z) {
    static double key_points[100][3];
    static bool initialized = false;
    
    if (!initialized) {
        generate_target_key_points(key_points);
        initialized = true;
    }
    
    *x = key_points[0][0];
    *y = key_points[0][1];
    *z = key_points[0][2];
}

// 性能测试函数
EXPORT double performance_test(int iterations) {
    double test_missile[3] = {10000.0, 0.0, 1000.0};
    double test_cloud[3] = {5000.0, 100.0, 500.0};
    
    clock_t start = clock();
    
    for (int i = 0; i < iterations; i++) {
        check_complete_blocking(test_missile, test_cloud);
    }
    
    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

// 简单测试函数
EXPORT int simple_test() {
    double missile[3] = {0.0, 0.0, 1000.0};
    double cloud[3] = {0.0, 200.0, 500.0};
    
    bool result = check_complete_blocking(missile, cloud);
    return result ? 1 : 0;
}

// 调试信息函数
EXPORT void debug_info() {
    printf("C库调试信息:\n");
    printf("  目标关键点数量: %d\n", NUM_KEY_POINTS);
    printf("    - 底面圆周点: %d\n", NUM_BOTTOM_POINTS);
    printf("    - 顶面圆周点: %d\n", NUM_TOP_POINTS);
    printf("    - 侧面分层点: %d\n", NUM_SIDE_POINTS);
    printf("  烟雾有效半径: %.1f m\n", R_SMOKE);
    printf("  真目标中心: (%.1f, %.1f, %.1f)\n", 
           REAL_TARGET_CENTER[0], REAL_TARGET_CENTER[1], REAL_TARGET_CENTER[2]);
    printf("  真目标半径: %.1f m\n", REAL_TARGET_RADIUS);
    printf("  真目标高度: %.1f m\n", REAL_TARGET_HEIGHT);
}

// 多烟雾弹测试函数
EXPORT int test_multiple_clouds() {
    // 测试场景：2个烟雾弹
    double missile[3] = {0.0, 0.0, 1000.0};
    double clouds[2][3] = {
        {0.0, 190.0, 500.0},  // 第一个烟雾弹
        {0.0, 210.0, 500.0}   // 第二个烟雾弹
    };
    
    bool result = check_multiple_clouds_blocking(missile, clouds, 2);
    return result ? 1 : 0;
}

// 获取采样点详细信息
EXPORT void get_sampling_info(int* total_points, int* bottom_points, int* top_points, int* side_points) {
    *total_points = NUM_KEY_POINTS;
    *bottom_points = NUM_BOTTOM_POINTS;
    *top_points = NUM_TOP_POINTS;
    *side_points = NUM_SIDE_POINTS;
}