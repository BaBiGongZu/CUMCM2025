"""
使用PSO（粒子群优化）求解无人机FY2和FY3单独的最优策略
复用solve2_23.py中的物理模型和C库接口
"""

import numpy as np
import pyswarms as ps
from scipy.stats import qmc
import time
import warnings
import math
import matplotlib.pyplot as plt

# 复用solve2_23.py中的所有组件
from solve2_23 import (
    # C库实例
    c_smoke_lib,

    # 常量
    G_ACCEL, P_M1_0, VEC_V_M1, MISSILE_FLIGHT_TIME,
    P_FY2_0, P_FY3_0, REAL_TARGET_HEIGHT, SMOKE_DURATION,
    OPTIMIZER_TIME_STEP, V_SMOKE_SINK_SPEED,

    # 函数
    calculate_uav_direction_from_angle,
    print_solution_details
)

# 全局变量用于PSO目标函数
current_uav_pos = None
current_uav_name = None

def pso_objective_function(particles):
    """
    PSO目标函数 - 适配粒子群的批量计算
    particles: shape (n_particles, 4) - [uav_speed, flight_angle, t_drop, t_explode_delay]
    返回: shape (n_particles,) 的适应度数组
    """
    n_particles = particles.shape[0]
    fitness_values = np.zeros(n_particles)

    for i, params in enumerate(particles):
        try:
            uav_speed, flight_angle, t_drop, t_explode_delay = params

            # 计算总起爆时间
            t_explode_abs = t_drop + t_explode_delay

            # 约束检查
            if t_explode_abs >= MISSILE_FLIGHT_TIME:
                fitness_values[i] = 1000.0  # PSO最小化，所以用大的惩罚值
                continue

            # 1. 计算无人机飞行方向
            uav_direction = calculate_uav_direction_from_angle(flight_angle)

            # 2. 计算烟雾弹投放位置
            uav_drop_pos = current_uav_pos + uav_speed * t_drop * uav_direction

            # 3. 计算烟雾弹起爆位置
            bomb_initial_velocity = uav_speed * uav_direction
            explode_pos = uav_drop_pos.copy()
            explode_pos += bomb_initial_velocity * t_explode_delay
            explode_pos[2] -= 0.5 * G_ACCEL * t_explode_delay**2

            # 检查起爆高度是否合理
            if explode_pos[2] <= REAL_TARGET_HEIGHT:
                fitness_values[i] = 1000.0
                continue

            # 4. 使用C库接口进行计算
            explode_positions = [explode_pos]
            explode_times = [t_explode_abs]

            # 调用C库函数
            total_duration = c_smoke_lib.calculate_total_duration(
                P_M1_0, VEC_V_M1, explode_positions, explode_times,
                MISSILE_FLIGHT_TIME, OPTIMIZER_TIME_STEP,
                V_SMOKE_SINK_SPEED, SMOKE_DURATION
            )

            # PSO最小化，所以返回负的遮蔽时长
            fitness_values[i] = -total_duration if total_duration > 0 else 1000.0

        except Exception:
            fitness_values[i] = 1000.0

    return fitness_values

def create_pso_bounds(uav_name):
    """为指定无人机创建PSO优化的边界"""
    if uav_name == "FY2":
        # FY2的标准边界
        lower_bounds = [70.0, 0.0, 0.0, 0.0]        # [speed, angle, t_drop, t_delay]
        upper_bounds = [140.0, 2*np.pi, 60.0, 60.0] # [speed, angle, t_drop, t_delay]
    else:  # FY3
        # FY3需要特殊处理：缩短延迟时间，减少地面引爆概率
        lower_bounds = [70.0, 0.0, 0.0, 0.0]        # [speed, angle, t_drop, t_delay]
        upper_bounds = [140.0, 2*np.pi, 60.0, 60.0]  # [speed, angle, t_drop, t_delay]

    return (np.array(lower_bounds), np.array(upper_bounds))

def initialize_swarm_with_heuristics(n_particles, bounds, uav_name):
    """
    使用启发式种子和拉丁超立方采样初始化粒子群
    """
    lower_bounds, upper_bounds = bounds
    n_dimensions = len(lower_bounds)

    # 根据无人机选择启发式种子
    if uav_name == "FY2":
        heuristic_seeds = [
            [90.0, 4.0, 2.0, 15.0],     # 向西北方向，长延迟
            [100.0, 3.5, 3.0, 12.0],    # 向西方向
            [110.0, 4.5, 1.5, 18.0],    # 向北方向，最长延迟
            [120.0, 3.8, 4.0, 10.0],    # 向西北偏西
            [85.0, 4.2, 2.5, 16.0],     # 向北偏西
            [95.0, 3.0, 3.5, 14.0],     # 向西南方向
            [105.0, 5.0, 1.0, 19.0],    # 向北方向
        ]
    else:  # FY3
        heuristic_seeds = [
            [120.0, 1.5708, 2.0, 1.5],    # 向北方向，短延迟
            [100.0, 1.2566, 1.5, 2.0],    # 向东北方向，很短延迟
            [140.0, 1.8326, 3.0, 1.0],    # 高速向北，最短延迟
            [90.0, 1.3963, 2.5, 2.5],     # 中速向东北
            [110.0, 1.7453, 1.0, 3.0],    # 早投放，稍长延迟
            [130.0, 1.0472, 4.0, 0.5],    # 晚投放，极短延迟
            [80.0, 2.0944, 2.0, 4.0],     # 低速，中等延迟
        ]

    # 确保所有种子都在边界范围内
    validated_seeds = []
    for seed in heuristic_seeds:
        validated_seed = np.clip(seed, lower_bounds, upper_bounds)
        validated_seeds.append(validated_seed)

    # 初始化粒子位置
    initial_positions = np.zeros((n_particles, n_dimensions))

    # 使用启发式种子
    num_seeds = min(len(validated_seeds), n_particles)
    for i in range(num_seeds):
        initial_positions[i] = validated_seeds[i]

    # 剩余粒子使用拉丁超立方采样
    num_random = n_particles - num_seeds
    if num_random > 0:
        sampler = qmc.LatinHypercube(d=n_dimensions, seed=42)
        random_init_unit_scale = sampler.random(n=num_random)

        if uav_name == "FY3":
            # 对FY3的随机种群进行特殊处理，偏向较短的延迟时间
            delay_indices = 3  # t_explode_delay的索引
            random_init_unit_scale[:, delay_indices] = random_init_unit_scale[:, delay_indices] ** 2

        scaled_random_init = qmc.scale(random_init_unit_scale, lower_bounds, upper_bounds)
        initial_positions[num_seeds:] = scaled_random_init

    # 最终验证所有粒子都在边界内
    initial_positions = np.clip(initial_positions, lower_bounds, upper_bounds)

    print(f"完成{uav_name}粒子群初始化: {num_seeds} 个启发式种子 + {num_random} 个拉丁超立方采样粒子")

    return initial_positions

def optimize_single_uav_pso(uav_name, uav_initial_pos):
    """使用PSO优化单个无人机的策略"""
    global current_uav_pos, current_uav_name

    print(f"\n🎯 开始PSO优化{uav_name}单独策略")
    print("=" * 60)
    print(f"{uav_name}初始位置: [{uav_initial_pos[0]:.1f}, {uav_initial_pos[1]:.1f}, {uav_initial_pos[2]:.1f}]")

    # 设置全局变量
    current_uav_pos = uav_initial_pos.copy()
    current_uav_name = uav_name

    # 设置PSO参数
    n_particles = 1000
    n_dimensions = 4  # [speed, angle, t_drop, t_delay]
    max_iter = 2000

    # 创建边界
    bounds = create_pso_bounds(uav_name)

    # 使用拉丁超立方采样初始化粒子群
    initial_positions = initialize_swarm_with_heuristics(n_particles, bounds, uav_name)

    # 设置PSO超参数
    options = {
        'c1': 2.0,      # 认知参数（个体最优）
        'c2': 2.0,      # 社会参数（全局最优）
        'w': 0.9        # 惯性权重
    }

    print(f"\nPSO参数设置:")
    print(f"  粒子数量: {n_particles}")
    print(f"  最大迭代次数: {max_iter}")
    print(f"  认知参数c1: {options['c1']}")
    print(f"  社会参数c2: {options['c2']}")
    print(f"  惯性权重w: {options['w']}")

    if uav_name == "FY3":
        print("  > FY3特殊处理: 缩短延迟时间边界和偏向短延迟的随机分布")

    # 测试几个种子的有效性
    print("测试启发式种子有效性:")
    test_seeds = initial_positions[:min(3, len(initial_positions))]
    valid_seeds = 0
    for i, seed in enumerate(test_seeds):
        test_value = pso_objective_function(np.array([seed]))[0]
        if test_value < 100.0:  # 小于惩罚值
            valid_seeds += 1
        print(f"  种子{i+1}: {-test_value:.6f}")

    print(f"有效种子数量: {valid_seeds}/{len(test_seeds)}")

    # 创建优化器
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=n_dimensions,
        options=options,
        bounds=bounds,
        init_pos=initial_positions
    )

    print("开始PSO优化...")
    start_time = time.time()

    # 执行优化
    best_cost, best_pos = optimizer.optimize(
        pso_objective_function,
        iters=max_iter,
        verbose=True
    )

    end_time = time.time()
    print(f"\n{uav_name} PSO优化完成，耗时: {end_time - start_time:.2f} 秒")

    if best_cost < 100.0:  # 小于惩罚值说明找到有效解
        max_duration = -best_cost
        print_solution_details(best_pos, max_duration, uav_name, uav_initial_pos)

        print(f"\n{uav_name} PSO最优参数:")
        seed_string = ", ".join([f"{val:.6f}" for val in best_pos])
        print(f"[{seed_string}]")

        return best_pos, max_duration
    else:
        print(f"\n❌ {uav_name} PSO优化未找到有效解")
        print(f"最佳成本值: {best_cost:.6f}")
        print(f"最佳参数: {best_pos}")

        # 即使没有找到最优解，也返回最佳尝试
        if best_cost < 1000.0:
            return best_pos, -best_cost
        return None, 0.0

def run_pso_optimization():
    """运行PSO优化主程序"""
    print("🎯 FY2和FY3无人机单独最优策略求解 (PSO版本)")
    print("=" * 60)
    print(f"导弹总飞行时间: {MISSILE_FLIGHT_TIME:.2f} s")
    print(f"时间步长: {OPTIMIZER_TIME_STEP:.3f} s")

    # 选择性运行配置
    SOLVE_FY2 = False   # 设为False跳过FY2优化
    SOLVE_FY3 = True    # 设为False跳过FY3优化

    print(f"\n运行配置: FY2={'启用' if SOLVE_FY2 else '跳过'}, FY3={'启用' if SOLVE_FY3 else '跳过'}")

    fy2_params, fy2_duration = None, 0.0
    fy3_params, fy3_duration = None, 0.0

    # 优化FY2
    if SOLVE_FY2:
        fy2_params, fy2_duration = optimize_single_uav_pso("FY2", P_FY2_0)
    else:
        print("\n⏭️  跳过FY2优化")

    # 优化FY3
    if SOLVE_FY3:
        fy3_params, fy3_duration = optimize_single_uav_pso("FY3", P_FY3_0)
    else:
        print("\n⏭️  跳过FY3优化")

    # 汇总结果
    print("\n" + "=" * 80)
    print("📊 PSO单独策略汇总结果:")
    print("=" * 80)

    if SOLVE_FY2:
        if fy2_params is not None:
            print(f"FY2最优遮蔽时长: {fy2_duration:.6f} 秒")
            fy2_seed = ", ".join([f"{val:.6f}" for val in fy2_params])
            print(f"FY2最优参数: [{fy2_seed}]")
        else:
            print("FY2: 未找到有效解")
    else:
        print("FY2: 跳过优化")

    print()

    if SOLVE_FY3:
        if fy3_params is not None:
            print(f"FY3最优遮蔽时长: {fy3_duration:.6f} 秒")
            fy3_seed = ", ".join([f"{val:.6f}" for val in fy3_params])
            print(f"FY3最优参数: [{fy3_seed}]")
        else:
            print("FY3: 未找到有效解")
    else:
        print("FY3: 跳过优化")

    print("=" * 80)

    return fy2_params, fy2_duration, fy3_params, fy3_duration

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 运行PSO优化
    fy2_params, fy2_duration, fy3_params, fy3_duration = run_pso_optimization()

    print("\n🎯 PSO优化程序结束")

