"""
问题四：使用PSO（粒子群优化）解决三架无人机协同烟幕干扰问题
复用solve4_lib.py中的物理模型和C库接口
"""

import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import time
import warnings
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# 从solve4_lib.py导入所需的模块
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 复用solve4_lib.py中的常量和函数
from solve4_lib import (
    # 常量
    G_ACCEL, P_M1_0, VEC_V_M1, MISSILE_FLIGHT_TIME,
    P_FY1_0, P_FY2_0, P_FY3_0, UAV_POSITIONS,
    REAL_TARGET_HEIGHT, SMOKE_DURATION, OPTIMIZER_TIME_STEP,
    NUM_UAVS, V_SMOKE_SINK_SPEED, TARGET_FALSE,

    # C库实例
    c_smoke_lib,

    # 函数
    calculate_uav_direction_from_angle,
    decode_params_to_trajectories,
    print_solution_details,
    visualize_solution
)

# 添加拉丁超立方采样的导入
from scipy.stats import qmc

def pso_objective_function(particles):
    """
    PSO目标函数 - 适配粒子群的批量计算
    particles: shape (n_particles, n_dimensions)
    返回: shape (n_particles,) 的适应度数组
    """
    n_particles = particles.shape[0]
    fitness_values = np.zeros(n_particles)

    for i, params in enumerate(particles):
        try:
            trajectories = decode_params_to_trajectories(params)

            explode_positions = []
            explode_times = []

            # 约束检查
            valid = True
            for traj in trajectories:
                if traj['t_explode'] >= MISSILE_FLIGHT_TIME or traj['explode_pos'][2] <= REAL_TARGET_HEIGHT:
                    valid = False
                    break
                explode_positions.append(traj['explode_pos'])
                explode_times.append(traj['t_explode'])

            if not valid:
                fitness_values[i] = 1000.0  # PSO最小化，所以用大的惩罚值
                continue

            # 调用C库计算总遮蔽时长
            duration = c_smoke_lib.calculate_total_duration(
                P_M1_0, VEC_V_M1,
                explode_positions, explode_times,
                MISSILE_FLIGHT_TIME, OPTIMIZER_TIME_STEP,
                V_SMOKE_SINK_SPEED, SMOKE_DURATION
            )

            # PSO最小化，所以返回负的遮蔽时长
            fitness_values[i] = -duration if duration > 0 else 1000.0

        except Exception:
            fitness_values[i] = 1000.0

    return fitness_values

def create_pso_bounds():
    """创建PSO优化的边界"""
    # 参数边界: [speed1, angle1, t_drop1, t_delay1, speed2, angle2, t_drop2, t_delay2, speed3, angle3, t_drop3, t_delay3]
    lower_bounds = [
        # FY1 参数
        70.0, 0.0, 0.0, 0.0,        # speed1, angle1, t_drop1, t_delay1
        # FY2 参数
        70.0, 0.0, 0.0, 0.0,        # speed2, angle2, t_drop2, t_delay2
        # FY3 参数
        70.0, 0.0, 0.0, 0.0         # speed3, angle3, t_drop3, t_delay3
    ]

    upper_bounds = [
        # FY1 参数
        140.0, 2*np.pi, 20.0, 20.0,  # speed1, angle1, t_drop1, t_delay1
        # FY2 参数
        140.0, 2*np.pi, 20.0, 20.0,  # speed2, angle2, t_drop2, t_delay2
        # FY3 参数
        140.0, 2*np.pi, 20.0, 20.0   # speed3, angle3, t_drop3, t_delay3
    ]

    return (np.array(lower_bounds), np.array(upper_bounds))

def initialize_swarm_with_heuristics(n_particles, bounds):
    """
    使用启发式种子和拉丁超立方采样初始化粒子群
    复用solve4_lib.py中的方法
    """
    print(f"初始化粒子群，粒子数量: {n_particles}")
    lower_bounds, upper_bounds = bounds
    n_dimensions = len(lower_bounds)

    # 启发式种子（从solve4_lib.py复用）
    heuristic_seeds = [
        [123.301369, 0.087085, 0.409919, 0.243292, 133.058918, 5.428318, 9.773231, 3.907213, 139.061933, 1.485688, 18.721434, 3.674324],
        [130.0, 3.14, 2.0, 5.0, 120.0, 1.57, 3.0, 6.0, 110.0, 4.71, 4.0, 7.0],
        [140.0, 0.0, 1.0, 4.0, 100.0, 2.35, 2.5, 5.5, 115.0, 5.50, 3.5, 6.5],
        [70.0, 0.5, 0.1, 19.0, 75.0, 2.0, 0.2, 18.5, 80.0, 3.5, 0.3, 19.5],
        [139.0, 1.0, 14.5, 0.5, 135.0, 4.0, 14.8, 0.8, 130.0, 5.5, 14.9, 1.0],
        [70.1, 6.28, 7.5, 10.0, 139.9, 0.01, 8.0, 9.5, 100.0, 3.14159, 6.0, 12.0],
        [90.0, 3.14159, 12.0, 2.0, 85.0, 4.71, 11.0, 3.0, 95.0, 1.57, 10.0, 4.0],
        [105.0, 0.78539, 5.0, 15.0, 110.0, 2.35619, 6.0, 14.0, 115.0, 5.49779, 7.0, 13.0],
        [120.0, 1.5, 1.0, 8.0, 125.0, 1.6, 1.1, 8.5, 130.0, 1.7, 1.2, 9.0],
        [80.0, 0.0, 2.0, 5.0, 100.0, 3.14, 8.0, 10.0, 120.0, 6.28, 14.0, 15.0],
        [110.0, 2.0, 3.0, 0.1, 115.0, 2.1, 4.0, 10.0, 120.0, 2.2, 5.0, 19.9],
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

    # 剩余粒子使用拉丁超立方采样（复用solve4_lib.py的方法）
    num_random = n_particles - num_seeds
    if num_random > 0:
        print(f"使用拉丁超立方采样生成 {num_random} 个随机粒子...")

        # 使用拉丁超立方采样生成剩余粒子
        sampler = qmc.LatinHypercube(d=n_dimensions, seed=42)
        random_init_unit_scale = sampler.random(n=num_random)
        scaled_random_init = qmc.scale(random_init_unit_scale, lower_bounds, upper_bounds)

        # 填入剩余粒子位置
        initial_positions[num_seeds:] = scaled_random_init

    print(f"完成粒子群初始化: {num_seeds} 个启发式种子 + {num_random} 个拉丁超立方采样粒子")

    # 最终验证所有粒子都在边界内
    initial_positions = np.clip(initial_positions, lower_bounds, upper_bounds)

    return initial_positions

def run_pso_optimization():
    """运行PSO优化"""
    print("🎯 问题四：使用PSO优化三架无人机协同烟幕干扰")
    print(f"导弹总飞行时间: {MISSILE_FLIGHT_TIME:.2f} s")
    print("无人机位置:")
    for i, pos in enumerate(UAV_POSITIONS):
        print(f"  FY{i+1}: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")

    # 设置PSO参数 - 调整为更合理的值
    n_particles = 3000  # 与solve4_lib.py保持一致
    n_dimensions = 12   # 3架无人机 × 4个参数
    max_iter = 3000     # 与solve4_lib.py保持一致

    # 创建边界
    bounds = create_pso_bounds()

    # 使用拉丁超立方采样初始化粒子群
    initial_positions = initialize_swarm_with_heuristics(n_particles, bounds)

    # 设置PSO超参数 - 调整为更标准的值
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

    # 创建优化器（使用全局PSO）
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=n_dimensions,
        options=options,
        bounds=bounds,
        init_pos=initial_positions
    )

    print("\n开始PSO优化...")
    start_time = time.time()

    # 执行优化
    best_cost, best_pos = optimizer.optimize(
        pso_objective_function,
        iters=max_iter,
        verbose=True
    )

    end_time = time.time()
    print(f"\n优化完成，总耗时: {end_time - start_time:.2f} 秒")

    # 处理结果
    if best_cost < 100.0:  # 小于惩罚值说明找到有效解
        max_duration = -best_cost
        print(f"\n🎯 PSO优化成功!")
        print_solution_details(best_pos, max_duration)

        print("\n" + "="*25 + " PSO最优参数 " + "="*25)
        seed_string = ", ".join([f"{val:.6f}" for val in best_pos])
        print(f"[{seed_string}]")
        print("="*70)

        return best_pos, max_duration
    else:
        print(f"\n❌ PSO优化未找到有效解")
        print(f"最佳成本值: {best_cost:.6f}")
        print("\n" + "="*25 + " PSO当前最佳参数 " + "="*25)
        seed_string = ", ".join([f"{val:.6f}" for val in best_pos])
        print(f"[{seed_string}]")
        print("="*70)

        return None, 0.0

def compare_with_differential_evolution():
    """与差分进化算法结果进行比较"""
    print("\n" + "="*50)
    print("🔍 算法性能对比分析")
    print("="*50)

    # DE算法的已知最优结果（从solve4_lib.py）
    de_best_params = [123.301369, 0.087085, 0.409919, 0.243292, 133.058918, 5.428318, 9.773231, 3.907213, 139.061933, 1.485688, 18.721434, 3.674324]

    # 计算DE结果的目标函数值
    de_fitness = -pso_objective_function(np.array([de_best_params]))[0]

    print(f"差分进化算法参考结果:")
    print(f"  遮蔽时长: {de_fitness:.6f} 秒")
    print(f"  参数: {[f'{x:.3f}' for x in de_best_params]}")

    return de_fitness

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 运行PSO优化
    pso_params, pso_duration = run_pso_optimization()

    # 与DE算法比较
    de_duration = compare_with_differential_evolution()

    # 性能分析
    if pso_params is not None:
        print(f"\n📊 算法性能比较:")
        print(f"  PSO结果: {pso_duration:.6f} 秒")
        print(f"  DE参考:  {de_duration:.6f} 秒")

        if pso_duration > de_duration:
            improvement = ((pso_duration - de_duration) / de_duration) * 100
            print(f"  PSO改进: +{improvement:.2f}% 🎉")
        else:
            decline = ((de_duration - pso_duration) / de_duration) * 100
            print(f"  PSO表现: -{decline:.2f}% 📉")

        # 可视化最优解
        print("\n生成PSO最优解可视化...")
        # visualize_solution(pso_params)

    print("\n🎯 PSO优化程序结束")
