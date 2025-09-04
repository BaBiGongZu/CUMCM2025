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
        计算点到直线的距离
        
        Args:
            point: 点坐标 (烟雾球心)
            line_start: 直线起点 (导弹位置)
            line_end: 直线终点 (目标关键点)
        
        Returns:
            距离值
        """
        # 直线方向向量
        line_vec = line_end - line_start
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 1e-10:
            return np.linalg.norm(point - line_start)
        
        # 单位方向向量
        line_unit = line_vec / line_length
        
        # 点到直线起点的向量
        point_vec = point - line_start
        
        # 投影长度
        projection = np.dot(point_vec, line_unit)
        
        # 最近点在直线上的位置
        if projection < 0:
            # 最近点是直线起点
            closest_point = line_start
        elif projection > line_length:
            # 最近点是直线终点
            closest_point = line_end
        else:
            # 最近点在直线段内
            closest_point = line_start + projection * line_unit
        
        # 返回距离
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
        
        return blocked_count
    
    def is_target_blocked(self, missile_pos, smoke_center, threshold=1):
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
        
    def solve_gurobi_model(self, missile_speed=600.0, time_horizon=100, num_segments=1000):
        """
        使用Gurobi求解导弹干扰优化问题
        
        Args:
            missile_speed: 导弹飞行速度 (m/s)
            time_horizon: 时间范围 (s)
            num_segments: 时间分段数
        """
        print(f"\n使用Gurobi求解优化模型...")
        print(f"导弹速度: {missile_speed} m/s")
        print(f"导弹飞行时间: {self.missile_distance/missile_speed:.2f} s")
        
        # 创建模型
        model = gp.Model("missile_interference")
        model.setParam('OutputFlag', 1)
        model.setParam('MIPGap', 0.01)
        model.setParam('TimeLimit', 180)
        
        # 时间离散化
        dt = time_horizon / num_segments
        T = list(range(num_segments + 1))
        
        # 决策变量
        # 无人机飞行方向 (归一化)
        dx = model.addVar(lb=-1, ub=1, name="direction_x")
        dy = model.addVar(lb=-1, ub=1, name="direction_y") 
        # 强制水平飞行，z方向速度分量为0
        dz = model.addVar(lb=0, ub=0, name="direction_z")
        
        # 无人机速度
        v_uav = model.addVar(lb=self.uav_speed_min, ub=self.uav_speed_max, name="uav_speed")
        
        # 烟雾弹投放时间和起爆时间 (连续变量)
        t_drop = model.addVar(lb=0, ub=time_horizon-5, name="drop_time")
        t_explode = model.addVar(lb=0, ub=time_horizon, name="explode_time")
        
        # 辅助变量：每个时间段的遮蔽状态
        blocked = {}
        for t in T:
            blocked[t] = model.addVar(vtype=GRB.BINARY, name=f"blocked_{t}")
        
        # 约束条件
        # 1. 方向向量归一化 (由于dz=0, 约束变为dx^2 + dy^2 = 1)
        model.addConstr(dx*dx + dy*dy == 1, "unit_direction_horizontal")
        
        # 2. 起爆时间约束
        model.addConstr(t_explode >= t_drop, "explode_after_drop")
        model.addConstr(t_explode <= t_drop + 15, "explode_timing")
        
        # 3. 遮蔽效果约束
        for t in T:
            time_val = t * dt
            
            # 只有在起爆后且在有效时间内才可能有遮蔽
            # 使用Big-M方法线性化条件
            M = 10000  # 大数
            
            # 时间条件：time_val >= t_explode and time_val <= t_explode + effective_duration
            y1 = model.addVar(vtype=GRB.BINARY, name=f"time_valid_{t}")
            model.addConstr(time_val >= t_explode - M*(1-y1), f"time_start_{t}")
            model.addConstr(time_val <= t_explode + self.effective_duration + M*(1-y1), f"time_end_{t}")
            
            # 如果时间无效，则blocked[t] = 0
            model.addConstr(blocked[t] <= y1, f"blocked_time_limit_{t}")
            
            # 距离条件的简化处理
            # 这里使用启发式约束，实际距离计算过于复杂
            # 假设在特定时间窗口内有较高的遮蔽概率
            missile_flight_time = self.missile_distance / missile_speed
            
            if time_val > 0.7 * missile_flight_time and time_val < 0.95 * missile_flight_time:
                # 在导弹接近目标时更容易实现遮蔽
                pass
            else:
                # 其他时间遮蔽效果较差
                model.addConstr(blocked[t] <= 0.3, f"low_block_prob_{t}")
        
        # 目标函数：最大化总遮蔽时间
        total_blocked_time = gp.quicksum(blocked[t] * dt for t in T)
        model.setObjective(total_blocked_time, GRB.MAXIMIZE)
        
        # 求解
        model.optimize()
        
        # 提取结果
        if model.status == GRB.OPTIMAL:
            solution = {
                'status': 'optimal',
                'uav_direction': np.array([dx.x, dy.x, dz.x]),
                'uav_speed': v_uav.x,
                'drop_time': t_drop.x,
                'explode_time': t_explode.x,
                'total_blocked_time': model.objVal,
                'objective_value': model.objVal
            }
            
            # 计算派生结果
            solution.update(self._calculate_derived_results(solution, missile_speed))
            
            return solution
        else:
            print(f"优化失败，状态: {model.status}")
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
    
    def simulate_detailed_blocking(self, solution):
        """详细模拟遮蔽过程"""
        if solution['status'] != 'optimal':
            return {'blocked_time': 0, 'blocking_intervals': []}
        
        missile_speed = solution['missile_speed']
        explode_time = solution['explode_time']
        explode_pos = solution['explode_position']
        
        # 时间步长
        dt = 0.1
        total_time = solution['missile_flight_time']
        
        blocked_intervals = []
        total_blocked = 0
        current_blocked_start = None
        
        for t in np.arange(0, total_time, dt):
            # 导弹位置
            missile_pos = self.missile_pos + missile_speed * t * self.missile_direction
            
            # 烟雾云位置
            if t >= explode_time and t <= explode_time + self.effective_duration:
                cloud_pos = explode_pos.copy()
                cloud_pos[2] -= self.cloud_sink_speed * (t - explode_time)
                
                # 检查云团是否还在地面以上
                if cloud_pos[2] > 0:
                    # 计算距离
                    distance = np.linalg.norm(missile_pos - cloud_pos)
                    
                    if distance <= self.effective_radius:
                        if current_blocked_start is None:
                            current_blocked_start = t
                        total_blocked += dt
                    else:
                        if current_blocked_start is not None:
                            blocked_intervals.append((current_blocked_start, t))
                            current_blocked_start = None
                else:
                    # 云团落地
                    if current_blocked_start is not None:
                        blocked_intervals.append((current_blocked_start, t))
                        current_blocked_start = None
            else:
                if current_blocked_start is not None:
                    blocked_intervals.append((current_blocked_start, t))
                    current_blocked_start = None
        
        # 处理最后一个区间
        if current_blocked_start is not None:
            blocked_intervals.append((current_blocked_start, total_time))
        
        return {
            'blocked_time': total_blocked,
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
