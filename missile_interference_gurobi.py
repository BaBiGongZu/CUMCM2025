"""
导弹干扰优化模型 - Gurobi版本
专门针对单导弹单无人机单烟雾弹的场景优化
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import matplotlib.pyplot as plt


class MissileInterferenceGurobi:
    def __init__(self):
        # 场景参数
        self.fake_target = np.array([0.0, 0.0, 0.0])
        self.real_target_center = np.array([0.0, 200.0, 0.0])
        self.real_target_radius = 7.0
        self.real_target_height = 10.0
        
        # 真目标圆柱体上的8个关键点K
        self.target_key_points = np.array([
            [0, 207, 10],   # 顶面前
            [0, 193, 10],   # 顶面后
            [7, 200, 10],   # 顶面右
            [-7, 200, 10],  # 顶面左
            [0, 207, 0],    # 底面前
            [0, 193, 0],    # 底面后
            [7, 200, 0],    # 底面右
            [-7, 200, 0]    # 底面左
        ])
        
        # 导弹M1参数
        self.missile_pos = np.array([20000.0, 0.0, 2000.0])
        self.missile_target = self.fake_target
        self.missile_distance = np.linalg.norm(self.missile_pos - self.missile_target)
        self.missile_direction = (self.missile_target - self.missile_pos) / self.missile_distance
        
        # 无人机FY1参数
        self.uav_pos = np.array([17800.0, 0.0, 1800.0])
        self.uav_speed_min = 70.0
        self.uav_speed_max = 140.0
        
        # 烟雾弹参数
        self.gravity = 9.8
        self.cloud_sink_speed = 3.0  # 烟雾云团下沉速度
        self.effective_radius = 10.0  # 有效遮蔽半径
        self.effective_duration = 20.0  # 有效遮蔽时间
        
        print(f"导弹到假目标距离: {self.missile_distance:.2f} m")
        print(f"导弹飞行方向: {self.missile_direction}")
        print(f"真目标关键点数量: {len(self.target_key_points)}")
    
    def point_to_line_distance(self, point, line_start, line_end):
        """
        计算点到线段的距离 (优化版本)
        
        Args:
            point: 点坐标 (烟雾球心)
            line_start: 直线起点 (导弹位置)
            line_end: 直线终点 (目标关键点)
        
        Returns:
            距离值
        """
        # 定义线段向量
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_len_sq = np.dot(line_vec, line_vec)
        
        # 如果线段起点和终点重合，直接返回点到该点的距离
        if line_len_sq < 1e-14:
            return np.linalg.norm(point_vec)
            
        # 计算投影比例 t = dot(point_vec, line_vec) / |line_vec|^2
        # t 表示投影点在线段方向向量上的位置
        t = np.dot(point_vec, line_vec) / line_len_sq
        
        # 将 t 限制在 [0, 1] 区间内，找到线段上的最近点
        t = max(0, min(1, t))
        
        # 计算最近点坐标
        closest_point = line_start + t * line_vec
        
        # 返回点到最近点的距离
        return np.linalg.norm(point - closest_point)
    
    def check_target_blocking(self, missile_pos, smoke_center):
        """
        检查烟雾是否遮挡了导弹到真目标关键点的视线
        
        Args:
            missile_pos: 导弹位置
            smoke_center: 烟雾球心位置
        
        Returns:
            遮蔽的关键点数量
        """
        if smoke_center is None:
            return 0
        
        blocked_count = 0
        
        for key_point in self.target_key_points:
            # 计算烟雾球心到导弹-关键点连线的距离
            distance = self.point_to_line_distance(smoke_center, missile_pos, key_point)
            
            # 如果距离小于有效半径，则该关键点被遮挡
            if distance <= self.effective_radius:
                blocked_count += 1
            else:
                return 0  # 只要有一个关键点未被遮挡，立即返回0
        
        return blocked_count
    
    def is_target_blocked(self, missile_pos, smoke_center, threshold=8):
        """
        判断目标是否被有效遮挡
        
        Args:
            missile_pos: 导弹位置
            smoke_center: 烟雾球心位置
            threshold: 需要遮挡的最少关键点数量
        
        Returns:
            是否被遮挡 (True/False)
        """
        blocked_count = self.check_target_blocking(missile_pos, smoke_center)
        return blocked_count >= threshold
        
    def solve_gurobi_model(self, missile_speed=300.0, time_horizon=67, num_segments=670):
        """
        使用Gurobi求解导弹干扰优化问题 (离散时间 + 二进制决策变量版)
        
        Args:
            missile_speed: 导弹飞行速度 (m/s)
            time_horizon: 时间范围 (s)
            num_segments: 时间分段数 (时间点总数)
        """
        print(f"\n使用Gurobi求解优化模型 (离散时间 + 二进制决策变量版)...")
        missile_flight_time = self.missile_distance / missile_speed
        print(f"导弹速度: {missile_speed} m/s, 飞行时间: {missile_flight_time:.2f} s")
        
        model = gp.Model("missile_interference_discrete_binary")
        model.setParam('NonConvex', 2)
        model.setParam('TimeLimit', 3600)
        model.setParam('MIPGap', 0.1)
        #model.setParam('NoRelHeurTime', 300)

        dt = time_horizon / num_segments # dt = 0.1s
        T = list(range(num_segments + 1)) # T = [0, 1, ..., 670]
        
        # --- 1. 决策变量 (使用离散时间点和二进制选择) ---
        dx = model.addVar(lb=-1, ub=1, name="dx")
        dy = model.addVar(lb=-1, ub=1, name="dy")
        v_uav = model.addVar(lb=self.uav_speed_min, ub=self.uav_speed_max, name="v_uav")
        
        # 使用整数变量代表投放时间点
        t_drop_tick = model.addVar(vtype=GRB.INTEGER, lb=0, ub=num_segments, name="t_drop_tick")
        
        # 使用一组二进制变量 x_i 代表在哪个时间点(tick)起爆
        x_explode_tick = model.addVars(T, vtype=GRB.BINARY, name="x_explode_tick")

        # 将整数tick转换为连续时间
        t_drop = t_drop_tick * dt
        # t_explode 是一个由二进制变量构成的线性表达式
        t_explode = gp.quicksum(i * dt * x_explode_tick[i] for i in T)

        # 辅助变量，用于线性化 v_uav * t_explode
        uav_dist = model.addVar(lb=0, ub=self.uav_speed_max * time_horizon, name="uav_dist")

        # --- 2. 基础约束 ---
        model.addConstr(dx*dx + dy*dy == 1, "unit_direction")
        # 核心约束: 确保只有一个起爆时间
        model.addConstr(gp.quicksum(x_explode_tick[i] for i in T) == 1, "single_explode_time")
        # 起爆时间必须在投放时间之后
        model.addConstr(t_explode >= t_drop, "explode_after_drop")
        model.addQConstr(uav_dist == v_uav * t_explode, "uav_dist_constr")

        # --- 3. 派生变量 ---
        fall_time = t_explode - t_drop
        
        # 起爆位置表达式 (使用辅助变量 uav_dist，避免三变量乘积)
        explode_pos_x = self.uav_pos[0] + uav_dist * dx
        explode_pos_y = self.uav_pos[1] + uav_dist * dy
        explode_pos_z = self.uav_pos[2] - 0.5 * self.gravity * fall_time * fall_time

        # --- 4. 遮挡约束 (循环内) ---
        blocked = model.addVars(T, vtype=GRB.BINARY, name="blocked")
        is_active = model.addVars(T, vtype=GRB.BINARY, name="is_active")
        
        num_kp = self.target_key_points.shape[0]
        is_blocked_by_kp = model.addVars(T, num_kp, vtype=GRB.BINARY, name="is_blocked_by_kp")
        all_kp_blocked = model.addVars(T, vtype=GRB.BINARY, name="all_kp_blocked")
        
        cross_x = model.addVars(T, num_kp, lb=-GRB.INFINITY, name="cross_x")
        cross_y = model.addVars(T, num_kp, lb=-GRB.INFINITY, name="cross_y")
        cross_z = model.addVars(T, num_kp, lb=-GRB.INFINITY, name="cross_z")
        dist_sq_var = model.addVars(T, num_kp, lb=0.0, name="dist_sq_var")

        # 烟雾持续时间对应的tick数量
        effective_duration_ticks = int(self.effective_duration / dt)

        for i in T:
            time_val = i * dt
            
            # a. 定义烟雾激活状态 (使用二进制变量x_i)
            # is_active[i] = 1 如果爆炸发生在 [i - duration_ticks, i] 这个时间窗口内
            start_tick_for_sum = max(0, i - effective_duration_ticks)
            # 结束点是i，因为如果爆炸发生在i时刻，那么在i时刻烟雾是激活的
            relevant_x_vars = [x_explode_tick[j] for j in range(start_tick_for_sum, i + 1)]
            
            if relevant_x_vars:
                model.addConstr(is_active[i] == gp.quicksum(relevant_x_vars), f"is_active_def_{i}")
            else:
                # 对于i < duration_ticks的情况，求和范围可能为空
                model.addConstr(is_active[i] == 0, f"is_active_def_{i}")

            # b. 计算该时刻的导弹和烟雾位置 (作为表达式)
            missile_pos_t = self.missile_pos + missile_speed * time_val * self.missile_direction
            cloud_pos_x_t = explode_pos_x
            cloud_pos_y_t = explode_pos_y
            cloud_pos_z_t = explode_pos_z - self.cloud_sink_speed * (time_val - t_explode)
            
            # c. 对每个关键点，检查是否被遮挡 (使用点到直线距离)
            for k, key_point in enumerate(self.target_key_points):
                # 定义视线向量 (从导弹到关键点)
                line_start = missile_pos_t
                line_end = key_point
                line_vec = line_end - line_start
                line_len_sq = np.dot(line_vec, line_vec)
                
                if line_len_sq < 1e-9: 
                    model.addConstr(is_blocked_by_kp[i, k] == 0)
                    continue

                # 定义烟雾球心到视线起点的向量
                point_vec_x = cloud_pos_x_t - line_start[0]
                point_vec_y = cloud_pos_y_t - line_start[1]
                point_vec_z = cloud_pos_z_t - line_start[2]

                # 分解复杂的非线性表达式
                model.addConstr(cross_x[i, k] == point_vec_y * line_vec[2] - point_vec_z * line_vec[1])
                model.addConstr(cross_y[i, k] == point_vec_z * line_vec[0] - point_vec_x * line_vec[2])
                model.addConstr(cross_z[i, k] == point_vec_x * line_vec[1] - point_vec_y * line_vec[0])

                cross_prod_sq = cross_x[i, k]**2 + cross_y[i, k]**2 + cross_z[i, k]**2
                
                model.addQConstr(dist_sq_var[i, k] * line_len_sq >= cross_prod_sq, f"dist_sq_calc_{i}_{k}")

                model.addGenConstrIndicator(is_blocked_by_kp[i, k], True, 
                                            dist_sq_var[i, k] <= self.effective_radius**2)

            # d. 遮挡逻辑
            kp_vars = [is_blocked_by_kp[i, k] for k in range(self.target_key_points.shape[0])]
            model.addGenConstrAnd(all_kp_blocked[i], kp_vars, f"and_constr_all_kp_{i}")

            # 最终遮挡状态
            model.addGenConstrAnd(blocked[i], [is_active[i], all_kp_blocked[i]], f"and_constr_final_{i}")
            
        # --- 5. 目标函数 ---
        total_blocked_time = gp.quicksum(blocked[i] * dt for i in T)
        model.setObjective(total_blocked_time, GRB.MAXIMIZE)
        
        # --- 6. 求解 ---
        model.optimize()
        
        # --- 7. 提取结果 ---
        if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            solution = {
                'status': 'optimal' if model.status == GRB.OPTIMAL else 'suboptimal',
                'uav_direction': np.array([dx.x, dy.x, 0.0]),
                'uav_speed': v_uav.x,
                'drop_time': t_drop.x,
                'explode_time': t_explode.x, # t_explode.x 会自动计算出正确的值
                'objective_value': model.objVal
            }
            solution.update(self._calculate_derived_results(solution, missile_speed))
            return solution
        else:
            print(f"优化失败或未找到可行解，状态码: {model.status}")
            if model.status == GRB.INFEASIBLE:
                print("模型不可行，正在计算IIS...")
                model.computeIIS()
                model.write("missile_model.ilp")
                print("IIS已写入文件 missile_model.ilp")
            return {'status': 'failed'}

    def _calculate_derived_results(self, solution, missile_speed):
        """计算派生结果"""
        # 无人机在投放时的位置
        uav_drop_pos = (self.uav_pos + 
                       solution['uav_speed'] * solution['drop_time'] * solution['uav_direction'])
        
        # 烟雾弹的初始水平速度（继承无人机速度）
        bomb_horizontal_velocity = solution['uav_speed'] * solution['uav_direction']
        
        # 烟雾弹轨迹：投放后到起爆的时间
        fall_time = solution['explode_time'] - solution['drop_time']
        
        # 起爆位置计算
        explode_pos = uav_drop_pos.copy()
        explode_pos[0] += bomb_horizontal_velocity[0] * fall_time
        explode_pos[1] += bomb_horizontal_velocity[1] * fall_time
        explode_pos[2] -= 0.5 * self.gravity * fall_time**2  # 重力影响
        
        # 确保起爆位置在地面以上
        if explode_pos[2] < 0:
            explode_pos[2] = max(0, explode_pos[2])
        
        return {
            'missile_speed': missile_speed,
            'missile_flight_time': self.missile_distance / missile_speed,
            'uav_drop_position': uav_drop_pos,
            'explode_position': explode_pos,
            'fall_time': fall_time
        }
    
    def simulate_detailed_blocking(self, solution, block_threshold=1):
        """
        详细模拟遮蔽过程 (使用精确的视线遮挡判断)
        
        Args:
            solution: Gurobi求解的结果
            block_threshold: 触发“已遮蔽”状态所需的最少被遮挡关键点数量
        """
        if solution['status'] != 'optimal':
            return {'blocked_time': 0, 'blocking_intervals': []}
        
        missile_speed = solution['missile_speed']
        explode_time = solution['explode_time']
        explode_pos = solution['explode_position']
        
        dt = 0.1  # 仿真步长
        total_time = solution['missile_flight_time']
        
        blocked_intervals = []
        total_blocked_time = 0
        is_blocking = False
        block_start_time = 0
        
        for t in np.arange(0, total_time, dt):
            # 1. 计算当前时刻的导弹位置
            missile_pos = self.missile_pos + missile_speed * t * self.missile_direction
            
            # 2. 计算当前时刻的烟雾云中心位置
            cloud_pos = None
            if t >= explode_time and t <= explode_time + self.effective_duration:
                current_cloud_pos = explode_pos.copy()
                current_cloud_pos[2] -= self.cloud_sink_speed * (t - explode_time)
                if current_cloud_pos[2] > -self.effective_radius: # 只要烟雾球体还在地面以上
                    cloud_pos = current_cloud_pos

            # 3. 判断是否遮挡
            currently_blocked = False
            if cloud_pos is not None:
                blocked_points = self.check_target_blocking(missile_pos, cloud_pos)
                if blocked_points >= block_threshold:
                    currently_blocked = True

            # 4. 记录遮蔽时间和区间
            if currently_blocked and not is_blocking:
                is_blocking = True
                block_start_time = t
            elif not currently_blocked and is_blocking:
                is_blocking = False
                blocked_intervals.append((block_start_time, t))
                total_blocked_time += (t - block_start_time)

        # 处理仿真结束时仍在遮蔽的情况
        if is_blocking:
            end_time = min(total_time, explode_time + self.effective_duration)
            blocked_intervals.append((block_start_time, end_time))
            total_blocked_time += (end_time - block_start_time)
            
        return {
            'blocked_time': total_blocked_time,
            'blocking_intervals': blocked_intervals
        }
    
    def print_solution(self, solution):
        """打印解决方案的全部信息"""
        if solution.get('status') != 'optimal':
            print("❌ 优化失败或未找到可行解")
            return

        # 详细遮蔽分析
        blocking_result = self.simulate_detailed_blocking(solution)

        print("\n" + "="*25 + " 优化结果详情 " + "="*25)
        
        print("\n--- 核心决策变量 ---")
        print(f"  无人机飞行方向 (dx, dy, dz): ({solution['uav_direction'][0]:.4f}, {solution['uav_direction'][1]:.4f}, {solution['uav_direction'][2]:.4f})")
        print(f"  无人机飞行速度: {solution['uav_speed']:.2f} m/s")
        print(f"  烟雾弹投放时间: {solution['drop_time']:.2f} s")
        print(f"  烟雾弹起爆时间: {solution['explode_time']:.2f} s")

        print("\n--- 遮蔽效果评估 ---")
        print(f"  Gurobi目标函数值 (估算遮蔽时间): {solution.get('objective_value', 0):.2f} s")
        print(f"  仿真计算的总遮蔽时间: {blocking_result['blocked_time']:.2f} s")
        if blocking_result['blocking_intervals']:
            print("  遮蔽时间区间:")
            for i, (start, end) in enumerate(blocking_result['blocking_intervals']):
                print(f"    区间 {i+1}: 从 {start:.2f}s 到 {end:.2f}s (时长: {end-start:.2f}s)")
        else:
            print("  无有效遮蔽时间区间。")

        print("\n--- 派生关键信息 ---")
        print(f"  导弹飞行总时间: {solution['missile_flight_time']:.2f} s (速度: {solution['missile_speed']:.1f} m/s)")
        print(f"  烟雾弹自由落体时间: {solution['fall_time']:.2f} s")
        print(f"  投放位置 (x, y, z): ({solution['uav_drop_position'][0]:.1f}, {solution['uav_drop_position'][1]:.1f}, {solution['uav_drop_position'][2]:.1f})")
        print(f"  起爆位置 (x, y, z): ({solution['explode_position'][0]:.1f}, {solution['explode_position'][1]:.1f}, {solution['explode_position'][2]:.1f})")
        
        print("\n" + "="*60)
    
    def visualize_solution(self, solution, save_path=None):
        """可视化解决方案"""
        if solution['status'] != 'optimal':
            print("无法可视化失败的解决方案")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('导弹干扰优化解决方案可视化', fontsize=16, fontweight='bold')
        
        # 1. 3D轨迹图
        ax1 = fig.add_subplot(221, projection='3d')
        
        # 关键点
        ax1.scatter(*self.fake_target, color='red', s=150, marker='x', label='假目标', linewidth=3)
        ax1.scatter(*self.real_target_center, color='green', s=150, marker='o', label='真目标')
        ax1.scatter(*self.missile_pos, color='blue', s=100, marker='^', label='导弹初始位置')
        ax1.scatter(*self.uav_pos, color='orange', s=100, marker='s', label='无人机初始位置')
        
        # 导弹轨迹
        t_missile = np.linspace(0, solution['missile_flight_time'], 100)
        missile_traj = []
        for t in t_missile:
            pos = self.missile_pos + solution['missile_speed'] * t * self.missile_direction
            missile_traj.append(pos)
        missile_traj = np.array(missile_traj)
        ax1.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], 
                'b-', linewidth=3, label='导弹轨迹')
        
        # 无人机轨迹
        t_uav = np.linspace(0, min(solution['drop_time'] + 5, solution['missile_flight_time']), 50)
        uav_traj = []
        for t in t_uav:
            pos = self.uav_pos + solution['uav_speed'] * t * solution['uav_direction']
            uav_traj.append(pos)
        uav_traj = np.array(uav_traj)
        ax1.plot(uav_traj[:, 0], uav_traj[:, 1], uav_traj[:, 2], 
                'orange', linewidth=3, label='无人机轨迹')
        
        # 关键事件点
        ax1.scatter(*solution['uav_drop_position'], color='purple', s=120, 
                   marker='v', label='烟雾弹投放点')
        ax1.scatter(*solution['explode_position'], color='red', s=120, 
                   marker='*', label='烟雾弹起爆点')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_title('3D场景图')
        
        # 2. 俯视图 (XY平面)
        ax2 = axes[0, 1]
        ax2.scatter(self.fake_target[0], self.fake_target[1], color='red', s=150, marker='x', label='假目标')
        ax2.scatter(self.real_target_center[0], self.real_target_center[1], color='green', s=150, marker='o', label='真目标')
        ax2.scatter(self.missile_pos[0], self.missile_pos[1], color='blue', s=100, marker='^', label='导弹起点')
        ax2.scatter(self.uav_pos[0], self.uav_pos[1], color='orange', s=100, marker='s', label='无人机起点')
        
        # 轨迹投影
        ax2.plot(missile_traj[:, 0], missile_traj[:, 1], 'b-', linewidth=2, label='导弹轨迹')
        ax2.plot(uav_traj[:, 0], uav_traj[:, 1], 'orange', linewidth=2, label='无人机轨迹')
        
        # 关键点
        ax2.scatter(solution['uav_drop_position'][0], solution['uav_drop_position'][1], 
                   color='purple', s=80, marker='v', label='投放点')
        ax2.scatter(solution['explode_position'][0], solution['explode_position'][1], 
                   color='red', s=80, marker='*', label='起爆点')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.legend()
        ax2.set_title('俯视图 (XY平面)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 高度-时间图
        ax3 = axes[1, 0]
        ax3.plot(t_missile, missile_traj[:, 2], 'b-', linewidth=2, label='导弹高度')
        ax3.plot(t_uav, uav_traj[:, 2], 'orange', linewidth=2, label='无人机高度')
        
        # 关键时间点
        ax3.axvline(x=solution['drop_time'], color='purple', linestyle='--', alpha=0.7, label='投放时间')
        ax3.axvline(x=solution['explode_time'], color='red', linestyle='--', alpha=0.7, label='起爆时间')
        
        # 烟雾云高度
        t_cloud = np.arange(solution['explode_time'], 
                           min(solution['explode_time'] + self.effective_duration, solution['missile_flight_time']), 
                           0.1)
        cloud_heights = []
        for t in t_cloud:
            h = solution['explode_position'][2] - self.cloud_sink_speed * (t - solution['explode_time'])
            if h > 0:
                cloud_heights.append(h)
            else:
                break
        
        if cloud_heights:
            ax3.plot(t_cloud[:len(cloud_heights)], cloud_heights, 'gray', linewidth=2, alpha=0.7, label='烟雾云高度')
        
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('高度 (m)')
        ax3.legend()
        ax3.set_title('高度-时间曲线')
        ax3.grid(True, alpha=0.3)
        
        # 4. 遮蔽效果时间图
        ax4 = axes[1, 1]
        
        # 详细遮蔽分析
        blocking_result = self.simulate_detailed_blocking(solution)
        
        # 绘制遮蔽状态
        dt = 0.1
        times = np.arange(0, solution['missile_flight_time'], dt)
        blocking_status = np.zeros(len(times))
        
        for start, end in blocking_result['blocking_intervals']:
            start_idx = int(start / dt)
            end_idx = int(end / dt)
            blocking_status[start_idx:end_idx] = 1
        
        ax4.fill_between(times, blocking_status, alpha=0.4, color='red', label='遮蔽状态')
        ax4.plot(times, blocking_status, 'red', linewidth=1)
        
        # 标记关键时间点
        ax4.axvline(x=solution['explode_time'], color='red', linestyle='--', alpha=0.7, label='起爆时间')
        ax4.axvline(x=solution['explode_time'] + self.effective_duration, 
                   color='gray', linestyle='--', alpha=0.5, label='烟雾失效时间')
        
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('遮蔽状态')
        ax4.set_ylim(-0.1, 1.1)
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['未遮蔽', '遮蔽'])
        ax4.legend()
        ax4.set_title(f'遮蔽效果图 (总遮蔽时间: {blocking_result["blocked_time"]:.2f}s)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # print(f"\n📊 图形已保存到: {save_path}") # 移除此行
        
        plt.show()


def main():
    """主函数"""
    print("🎯 导弹干扰优化程序 - Gurobi版本")
    print("="*60)
    
    # 创建优化器
    optimizer = MissileInterferenceGurobi()
    
    # 求解优化问题
    try:
        # 固定导弹速度为 300 m/s
        missile_speed = 300.0  # m/s
        
        print(f"\n设定导弹速度: {missile_speed} m/s")
        solution = optimizer.solve_gurobi_model(missile_speed=missile_speed)
        
        if solution and solution['status'] == 'optimal':
            print(f"\n🏆 优化求解完成:")
            optimizer.print_solution(solution)
            
            # # 可视化解 (已禁用)
            # save_path = "results/figures/missile_interference_gurobi_300ms.png"
            # optimizer.visualize_solution(solution, save_path)
        else:
            print("❌ 未找到可行解或优化失败")
            
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
