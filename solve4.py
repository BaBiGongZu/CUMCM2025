import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import time
import warnings
import math

# --- 1. 定义全局常量 (已更新以包含3架无人机) ---
G_ACCEL = 9.8
P_M1_0 = np.array([20000.0, 0.0, 2000.0])
V_M1_SPEED = 300.0
TARGET_FALSE = np.array([0.0, 0.0, 0.0])

VEC_M1_TO_TARGET = TARGET_FALSE - P_M1_0
DIST_M1_TO_TARGET = np.linalg.norm(VEC_M1_TO_TARGET)
UNIT_VEC_M1 = VEC_M1_TO_TARGET / DIST_M1_TO_TARGET
VEC_V_M1 = V_M1_SPEED * UNIT_VEC_M1
MISSILE_FLIGHT_TIME = DIST_M1_TO_TARGET / V_M1_SPEED

# 3架无人机的初始位置
P_FY1_0 = np.array([17800.0, 0.0, 1800.0])
P_FY2_0 = np.array([17800.0, -200.0, 1800.0]) # 假设对称部署
P_FY3_0 = np.array([17800.0, 200.0, 1800.0])  # 假设对称部署
UAV_START_POSITIONS = [P_FY1_0, P_FY2_0, P_FY3_0]

REAL_TARGET_CENTER = np.array([0.0, 200.0, 0.0])
REAL_TARGET_RADIUS = 7.0
REAL_TARGET_HEIGHT = 10.0
R_SMOKE = 10.0
V_SMOKE_SINK_SPEED = 3.0
SMOKE_DURATION = 20.0
NUM_UAVS = 3

# 优化器参数
OPTIMIZER_TIME_STEP = 0.01

def _generate_target_key_points(num_points_per_circle=20, num_height_levels=10):
    """生成目标关键点, 在圆柱体侧面均匀取点"""
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
    """根据飞行角度计算无人机飞行方向"""
    uav_direction = np.zeros(3)
    uav_direction[0] = np.cos(flight_angle)
    uav_direction[1] = np.sin(flight_angle)
    return uav_direction

def objective_function_3_uavs(params):
    """
    3架无人机协同遮蔽的目标函数
    
    params: 12个变量
    [
     speed1, angle1, t_drop1, t_delay1,  # FY1
     speed2, angle2, t_drop2, t_delay2,  # FY2
     speed3, angle3, t_drop3, t_delay3   # FY3
    ]
    """
    grenades_info = []
    
    # 1. 解码12个参数，计算3枚烟幕弹的轨迹
    for i in range(NUM_UAVS):
        uav_params = params[i*4 : (i+1)*4]
        uav_speed, flight_angle, t_drop, t_explode_delay = uav_params
        
        t_explode_abs = t_drop + t_explode_delay
        if t_explode_abs >= MISSILE_FLIGHT_TIME:
            continue

        uav_start_pos = UAV_START_POSITIONS[i]
        uav_direction = calculate_uav_direction_from_angle(flight_angle)
        uav_velocity_vec = uav_speed * uav_direction
        
        uav_drop_pos = uav_start_pos + uav_velocity_vec * t_drop
        
        explode_pos = uav_drop_pos + uav_velocity_vec * t_explode_delay
        explode_pos[2] -= 0.5 * G_ACCEL * t_explode_delay**2
        
        # 约束：起爆高度必须高于目标
        if explode_pos[2] <= REAL_TARGET_HEIGHT:
            continue
            
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
    """打印3架无人机解的详细信息"""
    print(f"\n找到最优策略（3架无人机协同）：")
    print(f"  > 最大有效遮蔽时长: {duration:.6f} 秒")
    print("=" * 60)
    
    for i in range(NUM_UAVS):
        uav_params = params[i*4 : (i+1)*4]
        uav_speed, flight_angle, t_drop, t_explode_delay = uav_params
        
        t_explode_abs = t_drop + t_explode_delay
        
        uav_start_pos = UAV_START_POSITIONS[i]
        uav_direction = calculate_uav_direction_from_angle(flight_angle)
        uav_velocity_vec = uav_speed * uav_direction
        
        uav_drop_pos = uav_start_pos + uav_velocity_vec * t_drop
        explode_pos = uav_drop_pos + uav_velocity_vec * t_explode_delay
        explode_pos[2] -= 0.5 * G_ACCEL * t_explode_delay**2

        print(f"无人机 FY{i+1} (从 {uav_start_pos[:2]} 开始):")
        print(f"  飞行策略: 速度 {uav_speed:.2f} m/s, 角度 {math.degrees(flight_angle):.2f}°")
        print(f"  投放时间: {t_drop:.4f} s (起爆: {t_explode_abs:.4f} s)")
        print(f"  起爆位置: [{explode_pos[0]:.2f}, {explode_pos[1]:.2f}, {explode_pos[2]:.2f}]")
        print("-" * 30)
    print("=" * 60)

if __name__ == "__main__":
    print("🎯 问题四：3架无人机协同干扰优化")
    print(f"导弹总飞行时间: {MISSILE_FLIGHT_TIME:.2f} s")
    
    start_time = time.time()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # 优化变量边界: 12个变量 (3 UAVs * 4 params/UAV)
    bounds_per_uav = [
        (70.0, 140.0),      # uav_speed (m/s)
        (0, 2 * np.pi),     # flight_angle (rad)
        (0.1, 15.0),        # t_drop (s)
        (0.0, 20.0)         # t_explode_delay (s)
    ]
    bounds = bounds_per_uav * NUM_UAVS
    
    # 使用一个合理的猜测作为种子 (12维)
    seed_per_uav = [100.0, 0.1, 2.0, 1.0] # 速度, 角度, 投放时间, 延迟
    seed = np.array([
        seed_per_uav * NUM_UAVS
    ])
    
    # 生成初始种群
    TOTAL_POPSIZE = 1000 # 变量增多，需要更大的种群
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
        objective_function_3_uavs,
        bounds,
        init=full_init_population,
        strategy='best1bin',
        maxiter=2000, # 增加迭代次数
        popsize=50,   # 增加种群乘数因子
        tol=0.01,
        recombination=0.7,
        mutation=(0.5, 1.0),
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
        seed_string = ", ".join([f"{val:.8f}" for val in best_params])
        print(f"seed = np.array([[\n    {seed_string}\n]])")
        print("=" * 60)
    else:
        print("\n❌ 优化失败或未找到有效解")
        if hasattr(result, 'x') and result.x is not None:
            print(f"最佳找到值: {-result.fun:.6f} 秒")
            print("\n" + "=" * 20 + " 当前最佳种子 " + "=" * 20)
            seed_string = ", ".join([f"{val:.8f}" for val in result.x])
            print(f"seed = np.array([[\n    {seed_string}\n]])")
            print("=" * 60)