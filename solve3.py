import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import time
import warnings
import math

# --- 1. 定义全局常量 ---
G_ACCEL = 9.8
P_M1_0 = np.array([20000.0, 0.0, 2000.0])
V_M1_SPEED = 300.0
TARGET_FALSE = np.array([0.0, 0.0, 0.0])

VEC_M1_TO_TARGET = TARGET_FALSE - P_M1_0
DIST_M1_TO_TARGET = np.linalg.norm(VEC_M1_TO_TARGET)
UNIT_VEC_M1 = VEC_M1_TO_TARGET / DIST_M1_TO_TARGET
VEC_V_M1 = V_M1_SPEED * UNIT_VEC_M1
MISSILE_FLIGHT_TIME = DIST_M1_TO_TARGET / V_M1_SPEED

P_FY1_0 = np.array([17800.0, 0.0, 1800.0])
REAL_TARGET_CENTER = np.array([0.0, 200.0, 0.0])
REAL_TARGET_RADIUS = 7.0
REAL_TARGET_HEIGHT = 10.0
R_SMOKE = 10.0
V_SMOKE_SINK_SPEED = 3.0
SMOKE_DURATION = 20.0
NUM_GRENADES = 3
MIN_DROP_INTERVAL = 1.0

# 优化器参数
OPTIMIZER_TIME_STEP = 0.01 # 适当增大步长以加快计算速度


def _generate_target_key_points(num_points_per_circle=20, num_height_levels=10):
    """
    生成目标关键点, 在圆柱体侧面均匀取点
    
    Parameters:
    -----------
    num_points_per_circle : int
        每个高度层上圆周的点数
    num_height_levels : int
        高度层数
    """
    key_points = []
    angles = np.linspace(0, 2*np.pi, num_points_per_circle, endpoint=False)
    heights = np.linspace(0.0, REAL_TARGET_HEIGHT, num_height_levels)
    
    for h in heights:
        for angle in angles:
            key_points.append([
                REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * np.cos(angle),
                REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * np.sin(angle),
                h
            ])
    return np.array(key_points)

TARGET_KEY_POINTS = _generate_target_key_points(num_points_per_circle=10, num_height_levels=10)


def point_to_line_distance(point, line_start, line_end):
    """计算点到线段的距离"""
    line_vec = line_end - line_start
    line_length_sq = np.dot(line_vec, line_vec)
    if line_length_sq < 1e-10:
        return np.linalg.norm(point - line_start)
    point_vec = point - line_start
    projection = np.dot(point_vec, line_vec) / line_length_sq
    if projection < 0:
        return np.linalg.norm(point - line_start)
    elif projection > 1:
        return np.linalg.norm(point - line_end)
    else:
        closest_point = line_start + projection * line_vec
        return np.linalg.norm(point - closest_point)


def calculate_uav_direction_from_angle(flight_angle):
    """
    根据飞行角度计算无人机飞行方向
    
    Parameters:
    -----------
    flight_angle : float
        飞行角度（弧度），以X轴正向为0，逆时针为正。
    
    Returns:
    --------
    np.array : 3D方向向量
    """
    # 基准方向为X轴正向 [1, 0]。逆时针旋转flight_angle弧度后，
    # 新方向的坐标为 (cos(flight_angle), sin(flight_angle))
    
    uav_direction = np.zeros(3)
    uav_direction[0] = np.cos(flight_angle)
    uav_direction[1] = np.sin(flight_angle)
    
    return uav_direction


def objective_function_3_grenades(params):
    """
    多枚烟雾弹协同遮蔽的目标函数
    
    params: [uav_speed, flight_angle, 
             t_drop1, t_drop_interval2, t_drop_interval3,
             t_explode_delay1, t_explode_delay2, t_explode_delay3]
    """
    uav_speed, flight_angle, t_d1, t_di2, t_di3, t_ed1, t_ed2, t_ed3 = params
    
    # 1. 计算无人机和炸弹的运动学
    uav_direction = calculate_uav_direction_from_angle(flight_angle)
    uav_velocity_vec = uav_speed * uav_direction
    
    drop_times = [
        t_d1,
        t_d1 + t_di2,
        t_d1 + t_di2 + t_di3
    ]
    explode_delays = [t_ed1, t_ed2, t_ed3]
    
    grenades_info = []
    for i in range(NUM_GRENADES):
        t_drop = drop_times[i]
        t_explode_delay = explode_delays[i]
        
        t_explode_abs = t_drop + t_explode_delay
        if t_explode_abs >= MISSILE_FLIGHT_TIME:
            continue

        uav_drop_pos = P_FY1_0 + uav_velocity_vec * t_drop
        
        explode_pos = uav_drop_pos + uav_velocity_vec * t_explode_delay
        explode_pos[2] -= 0.5 * G_ACCEL * t_explode_delay**2
        
            
        grenades_info.append({
            'explode_pos': explode_pos,
            't_explode_abs': t_explode_abs
        })

    if not grenades_info:
        return 0.0

    # 2. 模拟遮蔽过程
    t_start = min(g['t_explode_abs'] for g in grenades_info)
    t_end = min(max(g['t_explode_abs'] for g in grenades_info) + SMOKE_DURATION, MISSILE_FLIGHT_TIME)
    
    if t_start >= t_end:
        return 0.0

    num_steps = int((t_end - t_start) / OPTIMIZER_TIME_STEP) + 1
    t_array = np.linspace(t_start, t_end, num_steps)
    valid_duration_steps = 0
    
    for t in t_array:
        current_missile_pos = P_M1_0 + t * VEC_V_M1
        
        active_clouds = []
        for g in grenades_info:
            time_since_explode = t - g['t_explode_abs']
            if 0 <= time_since_explode < SMOKE_DURATION:
                cloud_pos = g['explode_pos'].copy()
                cloud_pos[2] -= V_SMOKE_SINK_SPEED * time_since_explode
                if cloud_pos[2] > 0:
                    active_clouds.append(cloud_pos)
        
        if not active_clouds:
            continue

        all_keypoints_blocked = True
        for key_point in TARGET_KEY_POINTS:
            is_kp_blocked = False
            for cloud_pos in active_clouds:
                if point_to_line_distance(cloud_pos, current_missile_pos, key_point) <= R_SMOKE:
                    is_kp_blocked = True
                    break
            if not is_kp_blocked:
                all_keypoints_blocked = False
                break
        
        if all_keypoints_blocked:
            valid_duration_steps += 1
            
    return -valid_duration_steps * OPTIMIZER_TIME_STEP


def print_solution_details(params, duration):
    """打印3枚弹的解的详细信息"""
    uav_speed, flight_angle, t_d1, t_di2, t_di3, t_ed1, t_ed2, t_ed3 = params
    
    print(f"\n找到最优策略（3枚烟雾弹）：")
    print(f"  > 最大有效遮蔽时长: {duration:.6f} 秒")
    print("-" * 60)
    print(f"无人机飞行速度: {uav_speed:.4f} m/s")
    print(f"飞行角度: {flight_angle:.6f} 弧度 ({math.degrees(flight_angle):.2f}°)")
    print("-" * 60)
    
    drop_times = [t_d1, t_d1 + t_di2, t_d1 + t_di2 + t_di3]
    explode_delays = [t_ed1, t_ed2, t_ed3]
    
    uav_direction = calculate_uav_direction_from_angle(flight_angle)
    uav_velocity_vec = uav_speed * uav_direction

    for i in range(NUM_GRENADES):
        t_drop = drop_times[i]
        t_explode_delay = explode_delays[i]
        t_explode_abs = t_drop + t_explode_delay
        
        uav_drop_pos = P_FY1_0 + uav_velocity_vec * t_drop
        explode_pos = uav_drop_pos + uav_velocity_vec * t_explode_delay
        explode_pos[2] -= 0.5 * G_ACCEL * t_explode_delay**2

        print(f"烟雾弹 {i+1}:")
        print(f"  投放时间: {t_drop:.4f} s (起爆: {t_explode_abs:.4f} s)")
        print(f"  投放位置: [{uav_drop_pos[0]:.2f}, {uav_drop_pos[1]:.2f}, {uav_drop_pos[2]:.2f}]")
        print(f"  起爆位置: [{explode_pos[0]:.2f}, {explode_pos[1]:.2f}, {explode_pos[2]:.2f}]")
    print("-" * 60)


if __name__ == "__main__":
    print("🎯 问题三：3枚烟雾弹协同干扰优化")
    print(f"导弹总飞行时间: {MISSILE_FLIGHT_TIME:.2f} s")
    
    start_time = time.time()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # 优化变量边界: 8个变量
    bounds = [
        (70.0, 140.0),      # uav_speed (m/s)
        (0, 2 * np.pi),     # flight_angle (rad)
        (0.1, 10.0),        # t_drop1 (s)
        (MIN_DROP_INTERVAL, 10.0), # t_drop_interval2 (s)
        (MIN_DROP_INTERVAL, 10.0), # t_drop_interval3 (s)
        (0.0, 20.0),        # t_explode_delay1 (s)
        (0.0, 20.0),        # t_explode_delay2 (s)
        (0.0, 20.0)         # t_explode_delay3 (s)
    ]
    
    # 使用一个合理的猜测作为种子
    # 注意：由于角度定义已更改，旧种子中的角度值可能不再适用。
    # 例如，旧的-3.0弧度现在可能需要表示为 pi/2 附近的值。
    # 优化器仍会从这个点开始搜索。
    seed = np.array([
        [130.0, 0.1, 0, 1, 1, 0.0, 0.0, 0.0] # 将角度种子改为一个更直观的猜测(朝向Y轴正向)
    ])
    
    # 生成初始种群
    TOTAL_POPSIZE = 500 # 增加种群大小以应对更复杂的问题
    num_random_individuals = TOTAL_POPSIZE - len(seed)
    num_vars = len(bounds)
    
    sampler = qmc.LatinHypercube(d=num_vars, seed=42)
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]
    random_init = sampler.random(n=num_random_individuals)
    scaled_random_init = qmc.scale(random_init, lower_bounds, upper_bounds)
    
    full_init_population = np.vstack((seed, scaled_random_init))
    
    print(f"初始种群大小: {TOTAL_POPSIZE}, 变量数: {num_vars}")
    print("开始优化...")
    
    result = differential_evolution(
        objective_function_3_grenades,
        bounds,
        init=full_init_population,
        strategy='best1bin', # 使用探索性更强的策略
        maxiter=1000,
        popsize=30, # popsize是乘数因子
        tol=0.01,
        recombination=0.8,
        mutation=(0.7, 1.5),
        disp=True,
        workers=-1,
        seed=42
    )
    
    end_time = time.time()
    print(f"\n优化完成，总耗时: {end_time - start_time:.2f} 秒")
    
    if result.success and result.fun < -1e-9:
        best_params = result.x
        max_duration = -result.fun
        
        print_solution_details(best_params, max_duration)
        
        print("\n" + "=" * 20 + " 下一轮迭代种子 " + "=" * 20)
        seed_string = ", ".join([f"{val:.12f}" for val in best_params])
        print(f"seed = np.array([[\n    {seed_string}\n]])")
        print("=" * 60)
    else:
        print("\n❌ 优化失败或未找到有效解")
        if result.fun is not None:
            print(f"最佳找到值: {-result.fun:.6f} 秒")