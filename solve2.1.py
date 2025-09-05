import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import time
import warnings
import math

# --- 1. 定义全局常量（与solve1.py保持一致）---
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
V_SMOKE_SINK_SPEED = 3.0  # 下沉速度标量
SMOKE_DURATION = 20.0

# 优化器参数
OPTIMIZER_TIME_STEP = 0.001


def _generate_target_key_points(num_points_per_circle=50):
    """生成目标关键点（与solve1.py完全一致）"""
    key_points = []
    angles = np.linspace(0, 2*np.pi, num_points_per_circle, endpoint=False)
    
    # 底面圆
    for angle in angles:
        key_points.append([
            REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * np.cos(angle),
            REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * np.sin(angle),
            0.0
        ])
    
    # 顶面圆
    for angle in angles:
        key_points.append([
            REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * np.cos(angle),
            REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * np.sin(angle),
            REAL_TARGET_HEIGHT
        ])
    
    return np.array(key_points)


TARGET_KEY_POINTS = _generate_target_key_points(num_points_per_circle=50)


def point_to_line_distance(point, line_start, line_end):
    """计算点到线段的距离（与solve1.py完全一致）"""
    line_vec = line_end - line_start
    line_length_sq = np.dot(line_vec, line_vec)
    
    if line_length_sq < 1e-10:
        return np.linalg.norm(point - line_start)
    
    point_vec = point - line_start
    projection = np.dot(point_vec, line_vec) / line_length_sq
    
    if projection < 0:
        closest_point = line_start
    elif projection > 1:
        closest_point = line_end
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


def corrected_objective_function(params):
    """
    修正的目标函数，使用与solve1.py一致的严谨物理模型
    
    Parameters:
    -----------
    params : array-like
        [uav_speed, flight_angle, t_drop, t_explode_delay]
        - uav_speed: 无人机速度 (m/s)
        - flight_angle: 飞行角度 (弧度), 以X轴正向为0, 逆时针为正
        - t_drop: 投放时间 (s)
        - t_explode_delay: 投放后到起爆的延迟时间 (s)
    """
    uav_speed, flight_angle, t_drop, t_explode_delay = params
    
    # 计算总起爆时间
    t_explode_abs = t_drop + t_explode_delay
    
    # 如果起爆时间超过导弹飞行时间，无效
    if t_explode_abs >= MISSILE_FLIGHT_TIME:
        return 0.0
    
    # 1. 计算无人机飞行方向（基于X轴正向的绝对角度）
    uav_direction = calculate_uav_direction_from_angle(flight_angle)
    
    # 2. 计算烟雾弹投放位置（第一阶段：无人机飞行）
    uav_drop_pos = P_FY1_0 + uav_speed * t_drop * uav_direction
    
    # 3. 计算烟雾弹起爆位置（第二阶段：抛物运动）
    bomb_initial_velocity = uav_speed * uav_direction
    explode_pos = uav_drop_pos.copy()
    explode_pos += bomb_initial_velocity * t_explode_delay
    explode_pos[2] -= 0.5 * G_ACCEL * t_explode_delay**2
    
    # 检查起爆高度是否合理
    if explode_pos[2] <= REAL_TARGET_HEIGHT:
        return 0.0
    
    # 4. 模拟遮蔽过程
    t_start = t_explode_abs
    t_end = min(t_explode_abs + SMOKE_DURATION, MISSILE_FLIGHT_TIME)
    
    if t_start >= t_end:
        return 0.0
    
    # 时间采样
    num_steps_actual = int((t_end - t_start) / OPTIMIZER_TIME_STEP) + 1
    if num_steps_actual <= 0:
        return 0.0
    
    t_array = np.linspace(t_start, t_end, num_steps_actual)
    valid_duration_steps = 0
    
    for t in t_array:
        # 当前导弹位置
        current_missile_pos = P_M1_0 + t * VEC_V_M1
        
        # 当前烟雾云位置（考虑下沉）
        time_since_explode = t - t_explode_abs
        current_cloud_pos = explode_pos.copy()
        current_cloud_pos[2] -= V_SMOKE_SINK_SPEED * time_since_explode
        
        # 如果云团落地，停止计算
        if current_cloud_pos[2] <= 0:
            break
        
        # 检查是否所有关键点都被遮蔽
        all_blocked = True
        for key_point in TARGET_KEY_POINTS:
            dist = point_to_line_distance(current_cloud_pos, current_missile_pos, key_point)
            if dist > R_SMOKE:
                all_blocked = False
                break
        
        if all_blocked:
            valid_duration_steps += 1
    
    total_duration = valid_duration_steps * OPTIMIZER_TIME_STEP
    return -total_duration  # 负值用于最大化


def print_solution_details(params, duration):
    """打印解的详细信息"""
    uav_speed, flight_angle, t_drop, t_explode_delay = params
    t_explode_abs = t_drop + t_explode_delay
    
    uav_direction = calculate_uav_direction_from_angle(flight_angle)
    uav_drop_pos = P_FY1_0 + uav_speed * t_drop * uav_direction
    bomb_initial_velocity = uav_speed * uav_direction
    explode_pos = uav_drop_pos.copy()
    explode_pos += bomb_initial_velocity * t_explode_delay
    explode_pos[2] -= 0.5 * G_ACCEL * t_explode_delay**2
    
    print(f"\n找到最优策略（基于严谨物理模型）：")
    print(f"  > 最大有效遮蔽时长: {duration:.6f} 秒")
    print("-" * 60)
    print(f"  无人机飞行速度: {uav_speed:.4f} m/s")
    print(f"  飞行角度: {flight_angle:.6f} 弧度 ({math.degrees(flight_angle):.2f}°)")
    print(f"  受领任务后投放时间: {t_drop:.4f} s")
    print(f"  投放后起爆延迟: {t_explode_delay:.4f} s")
    print(f"  总起爆时间: {t_explode_abs:.4f} s")
    print("-" * 60)
    print(f"  无人机飞行方向: [{uav_direction[0]:.6f}, {uav_direction[1]:.6f}, {uav_direction[2]:.6f}]")
    print(f"  投放位置: [{uav_drop_pos[0]:.2f}, {uav_drop_pos[1]:.2f}, {uav_drop_pos[2]:.2f}]")
    print(f"  起爆位置: [{explode_pos[0]:.2f}, {explode_pos[1]:.2f}, {explode_pos[2]:.2f}]")
    print("-" * 60)


if __name__ == "__main__":
    print("🎯 修正版全局优化程序（基于严谨物理模型）")
    print("=" * 60)
    print(f"导弹总飞行时间: {MISSILE_FLIGHT_TIME:.2f} s")
    print("采用与solve1.py完全一致的物理建模方法")
    
    start_time = time.time()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # 优化变量边界
    bounds = [
        (70.0, 140.0),      # uav_speed (m/s)
        (0, 2 * np.pi),     # flight_angle (rad) - 0到2pi
        (0.1, 10.0),        # t_drop (s) - 合理的投放时间范围
        (0.1, 10.0)         # t_explode_delay (s) - 合理的起爆延迟
    ]
    
    # 注意：由于角度定义已更改，旧的种子可能不再适用或不再是优解。
    # 可以注释掉种子，让优化器从头开始搜索，或者根据新角度定义提供一个新种子。
    # 这里我们先注释掉旧种子。
    seed = np.array([[
         123.809213143700, 0.096794168049, 0.570658303297, 0.405527351181
    ]])
    
    # 生成初始种群
    TOTAL_POPSIZE = 150
    num_random_individuals = TOTAL_POPSIZE - 1
    num_vars = len(bounds)
    
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    sampler = qmc.LatinHypercube(d=num_vars)
    random_init_unit_scale = sampler.random(n=num_random_individuals)
    scaled_random_init = qmc.scale(random_init_unit_scale, lower_bounds, upper_bounds)
    
    full_init_population = np.vstack((seed, scaled_random_init))
    
    print(f"初始种群大小: {TOTAL_POPSIZE} (包含1个种子)")
    print("开始优化...")
    
    # 差分进化优化
    result = differential_evolution(
        corrected_objective_function,
        bounds,
        init=full_init_population,
        strategy='rand1bin',
        maxiter=500,  # 适当减少迭代次数以节省时间
        tol=0.01,
        recombination=0.7,
        mutation=(0.5, 1.0), # 稍微减小变异范围，进行精细搜索
        disp=True,
        workers=-1,
        seed=42  # 固定随机种子以便重现
    )
    
    end_time = time.time()
    print(f"\n优化完成，总耗时: {end_time - start_time:.2f} 秒")
    
    if result.success and result.fun < -1e-9:
        best_params = result.x
        max_duration = -result.fun
        
        print_solution_details(best_params, max_duration)
        
        # 生成下一轮优化的种子
        print("\n" + "=" * 20 + " 下一轮迭代种子 " + "=" * 20)
        print("复制以下参数用于进一步优化：")
        seed_string = ", ".join([f"{val:.12f}" for val in best_params])
        print(f"seed = np.array([[\n    {seed_string}\n]])")
        print("=" * 60)
        
        # 验证结果：使用相同参数在check.py中验证
        print(f"\n🔍 验证提示：")
        print(f"在check.py中使用以下参数验证结果：")
        print(f"  uav_speed={best_params[0]:.6f}")
        print(f"  flight_angle={best_params[1]:.6f}")
        print(f"  t_drop={best_params[2]:.6f}")
        print(f"  t_explode_delay={best_params[3]:.6f}")
        
    else:
        print("\n❌ 优化失败或未找到有效解")
        print(f"最佳找到值: {-result.fun:.6f} 秒")
        
        # 测试solve1.py的默认参数
        print("\n🧪 测试solve1.py的默认参数...")
        default_result = corrected_objective_function([120.0, 0.0, 1.5, 3.6])
        print(f"solve1.py默认参数的遮蔽时长: {-default_result:.6f} 秒")