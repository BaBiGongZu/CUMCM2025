"""
导弹干扰优化模型
使用Gurobi求解最优无人机飞行策略，最大化对导弹的遮蔽时间
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import matplotlib.pyplot as plt


class MissileInterferenceOptimizer:
    def __init__(self):
        # 场景参数
        self.fake_target = np.array([0.0, 0.0, 0.0])  # 假目标坐标
        self.real_target_center = np.array([0.0, 200.0, 0.0])  # 真目标圆柱体底面圆心
        self.real_target_radius = 7.0  # 真目标圆柱体半径
        self.real_target_height = 10.0  # 真目标圆柱体高度
        
        # 导弹参数
        self.missile_initial_pos = np.array([20000.0, 0.0, 2000.0])  # 导弹M1初始位置
        self.missile_speed = None  # 导弹速度（由距离和时间计算）
        
        # 无人机参数
        self.uav_initial_pos = np.array([17800.0, 0.0, 1800.0])  # 无人机FY1初始位置
        self.uav_speed_min = 70.0  # 无人机最小速度 m/s
        self.uav_speed_max = 140.0  # 无人机最大速度 m/s
        
        # 烟雾弹参数
        self.gravity = 9.8  # 重力加速度
        self.smoke_cloud_speed = 3.0  # 烟雾云团下沉速度 m/s
        self.effective_radius = 10.0  # 有效遮蔽半径
        self.effective_duration = 20.0  # 有效遮蔽时间 s
        
        # 计算导弹到假目标的距离和飞行时间
        self.missile_distance = np.linalg.norm(self.missile_initial_pos - self.fake_target)
        self.missile_direction = (self.fake_target - self.missile_initial_pos) / self.missile_distance
        
    def calculate_missile_speed(self, total_time):
        """根据总飞行时间计算导弹速度"""
        return self.missile_distance / total_time
        
    def missile_position_at_time(self, t, missile_speed):
        """计算导弹在时间t时的位置"""
        if t <= 0:
            return self.missile_initial_pos
        
        distance_traveled = missile_speed * t
        if distance_traveled >= self.missile_distance:
            return self.fake_target
        
        return self.missile_initial_pos + distance_traveled * self.missile_direction
    
    def smoke_bomb_trajectory(self, uav_pos, uav_velocity, drop_time):
        """计算烟雾弹的轨迹"""
        def position_at_time(t):
            if t < drop_time:
                return None  # 还未投放
            
            dt = t - drop_time
            # 烟雾弹继承无人机的水平速度，垂直方向受重力影响
            x = uav_pos[0] + uav_velocity[0] * dt
            y = uav_pos[1] + uav_velocity[1] * dt
            z = uav_pos[2] - 0.5 * self.gravity * dt**2
            
            return np.array([x, y, z])
        
        return position_at_time
    
    def smoke_cloud_position(self, explosion_pos, explosion_time, t):
        """计算烟雾云团在时间t的位置（起爆后匀速下沉）"""
        if t < explosion_time:
            return None
        
        dt = t - explosion_time
        return explosion_pos - np.array([0, 0, self.smoke_cloud_speed * dt])
    
    def is_missile_blocked(self, missile_pos, smoke_center):
        """判断导弹是否被烟雾遮挡"""
        if smoke_center is None:
            return False
        
        # 计算导弹到烟雾中心的距离
        distance = np.linalg.norm(missile_pos - smoke_center)
        return distance <= self.effective_radius
    
    def solve_optimization(self, time_horizon=100.0, time_steps=1000):
        """使用Gurobi求解优化问题"""
        
        # 创建Gurobi模型
        model = gp.Model("missile_interference")
        model.setParam('OutputFlag', 1)
        model.setParam('TimeLimit', 300)  # 5分钟时间限制
        
        # 时间离散化
        dt = time_horizon / time_steps
        time_points = [i * dt for i in range(time_steps + 1)]
        
        # 决策变量
        # 无人机飞行方向（单位向量）
        uav_direction_x = model.addVar(lb=-1, ub=1, name="uav_dir_x")
        uav_direction_y = model.addVar(lb=-1, ub=1, name="uav_dir_y")
        uav_direction_z = model.addVar(lb=-1, ub=1, name="uav_dir_z")
        
        # 无人机飞行速度
        uav_speed = model.addVar(lb=self.uav_speed_min, ub=self.uav_speed_max, name="uav_speed")
        
        # 烟雾弹投放时间
        drop_time = model.addVar(lb=0, ub=time_horizon, name="drop_time")
        
        # 烟雾弹起爆时间
        explosion_time = model.addVar(lb=0, ub=time_horizon, name="explosion_time")
        
        # 导弹总飞行时间
        missile_flight_time = model.addVar(lb=10, ub=time_horizon, name="missile_flight_time")
        
        # 辅助变量：每个时间点的遮蔽状态
        blocked = {}
        for i, t in enumerate(time_points):
            blocked[i] = model.addVar(vtype=GRB.BINARY, name=f"blocked_{i}")
        
        # 约束条件
        # 1. 无人机方向向量为单位向量
        model.addConstr(
            uav_direction_x * uav_direction_x + 
            uav_direction_y * uav_direction_y + 
            uav_direction_z * uav_direction_z == 1,
            "unit_direction"
        )
        
        # 2. 起爆时间必须大于投放时间
        model.addConstr(explosion_time >= drop_time, "explosion_after_drop")
        
        # 3. 烟雾弹必须在落地前起爆
        # 这需要根据烟雾弹轨迹计算落地时间，这里简化处理
        model.addConstr(explosion_time <= drop_time + 20, "explosion_before_ground")
        
        # 4. 定义每个时间点的遮蔽状态
        for i, t in enumerate(time_points):
            if t > 5:  # 给系统一些启动时间
                # 这里需要线性化遮蔽条件，简化为线性约束
                # 实际实现中需要更复杂的线性化技术
                pass
        
        # 目标函数：最大化遮蔽时间
        total_blocked_time = gp.quicksum(blocked[i] * dt for i in range(len(time_points)))
        model.setObjective(total_blocked_time, GRB.MAXIMIZE)
        
        # 求解
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            result = {
                'uav_direction': np.array([
                    uav_direction_x.x,
                    uav_direction_y.x,
                    uav_direction_z.x
                ]),
                'uav_speed': uav_speed.x,
                'drop_time': drop_time.x,
                'explosion_time': explosion_time.x,
                'missile_flight_time': missile_flight_time.x,
                'total_blocked_time': model.objVal,
                'status': 'optimal'
            }
            return result
        else:
            return {'status': 'failed', 'code': model.status}
    
    def solve_simplified(self):
        """简化版本的求解方法，使用启发式算法"""
        print("使用简化启发式算法求解...")
        
        # 假设导弹飞行时间
        missile_flight_time = 30.0  # 假设30秒飞到目标
        missile_speed = self.missile_distance / missile_flight_time
        
        print(f"导弹飞行距离: {self.missile_distance:.2f} m")
        print(f"导弹飞行时间: {missile_flight_time:.2f} s")
        print(f"导弹飞行速度: {missile_speed:.2f} m/s")
        
        # 最优策略：无人机朝向目标区域飞行
        target_direction = (self.real_target_center - self.uav_initial_pos)
        target_direction = target_direction / np.linalg.norm(target_direction)
        
        # 选择中等速度
        uav_speed = 100.0
        
        # 计算最佳投放时间和起爆时间
        # 目标：让烟雾云在导弹接近真目标时形成最大遮蔽
        
        # 导弹到达真目标附近的时间
        missile_to_real_target_distance = np.linalg.norm(
            self.missile_initial_pos - self.real_target_center
        )
        missile_to_real_target_time = missile_to_real_target_distance / missile_speed
        
        # 提前投放烟雾弹，让其在导弹接近时起爆
        drop_time = max(0, missile_to_real_target_time - 15)
        explosion_time = missile_to_real_target_time - 5
        
        # 计算投放点的无人机位置
        uav_pos_at_drop = self.uav_initial_pos + uav_speed * drop_time * target_direction
        
        # 计算烟雾弹轨迹和起爆点
        bomb_trajectory = self.smoke_bomb_trajectory(uav_pos_at_drop, uav_speed * target_direction, drop_time)
        explosion_pos = bomb_trajectory(explosion_time)
        
        if explosion_pos is None or explosion_pos[2] < 0:
            explosion_time = drop_time + 10  # 调整起爆时间
            explosion_pos = bomb_trajectory(explosion_time)
        
        # 计算遮蔽时间
        blocked_time = 0
        dt = 0.1
        for t in np.arange(explosion_time, missile_flight_time, dt):
            missile_pos = self.missile_position_at_time(t, missile_speed)
            smoke_pos = self.smoke_cloud_position(explosion_pos, explosion_time, t)
            
            if smoke_pos is not None and smoke_pos[2] > 0:  # 烟雾还未落地
                if self.is_missile_blocked(missile_pos, smoke_pos):
                    blocked_time += dt
            
            # 烟雾有效时间限制
            if t - explosion_time > self.effective_duration:
                break
        
        result = {
            'uav_direction': target_direction,
            'uav_speed': uav_speed,
            'drop_time': drop_time,
            'explosion_time': explosion_time,
            'explosion_position': explosion_pos,
            'uav_drop_position': uav_pos_at_drop,
            'missile_flight_time': missile_flight_time,
            'missile_speed': missile_speed,
            'total_blocked_time': blocked_time,
            'status': 'heuristic_solution'
        }
        
        return result
    
    def print_solution(self, result):
        """打印解决方案"""
        print("\n" + "="*60)
        print("导弹干扰优化解决方案")
        print("="*60)
        
        if result['status'] == 'failed':
            print(f"❌ 优化失败，状态码: {result['code']}")
            return
        
        print(f"求解状态: {result['status']}")
        print(f"\n无人机飞行参数:")
        print(f"  初始位置: {self.uav_initial_pos}")
        print(f"  飞行方向: {result['uav_direction']}")
        print(f"  飞行速度: {result['uav_speed']:.2f} m/s")
        
        print(f"\n烟雾弹参数:")
        print(f"  投放时间: {result['drop_time']:.2f} s")
        print(f"  起爆时间: {result['explosion_time']:.2f} s")
        if 'uav_drop_position' in result:
            print(f"  投放位置: {result['uav_drop_position']}")
        if 'explosion_position' in result:
            print(f"  起爆位置: {result['explosion_position']}")
        
        print(f"\n导弹参数:")
        if 'missile_speed' in result:
            print(f"  飞行速度: {result['missile_speed']:.2f} m/s")
        print(f"  总飞行时间: {result['missile_flight_time']:.2f} s")
        
        print(f"\n优化结果:")
        print(f"  总遮蔽时间: {result['total_blocked_time']:.2f} s")
        
    def visualize_scenario(self, result, save_path=None):
        """可视化场景"""
        fig = plt.figure(figsize=(15, 10))
        
        # 3D图
        ax1 = fig.add_subplot(221, projection='3d')
        
        # 绘制目标
        ax1.scatter(*self.fake_target, color='red', s=100, label='假目标')
        ax1.scatter(*self.real_target_center, color='green', s=100, label='真目标')
        
        # 绘制初始位置
        ax1.scatter(*self.missile_initial_pos, color='blue', s=100, label='导弹初始位置')
        ax1.scatter(*self.uav_initial_pos, color='orange', s=100, label='无人机初始位置')
        
        # 绘制轨迹
        if result['status'] != 'failed':
            # 导弹轨迹
            missile_speed = result.get('missile_speed', 500)
            t_points = np.linspace(0, result['missile_flight_time'], 100)
            missile_traj = [self.missile_position_at_time(t, missile_speed) for t in t_points]
            missile_traj = np.array(missile_traj)
            ax1.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], 
                    'b-', linewidth=2, label='导弹轨迹')
            
            # 无人机轨迹
            uav_traj_time = min(result['drop_time'] + 5, result['missile_flight_time'])
            t_points = np.linspace(0, uav_traj_time, 100)
            uav_positions = []
            for t in t_points:
                pos = self.uav_initial_pos + result['uav_speed'] * t * result['uav_direction']
                uav_positions.append(pos)
            uav_positions = np.array(uav_positions)
            ax1.plot(uav_positions[:, 0], uav_positions[:, 1], uav_positions[:, 2], 
                    'orange', linewidth=2, label='无人机轨迹')
            
            # 标记关键点
            if 'uav_drop_position' in result:
                ax1.scatter(*result['uav_drop_position'], color='purple', s=80, 
                          marker='^', label='烟雾弹投放点')
            
            if 'explosion_position' in result:
                ax1.scatter(*result['explosion_position'], color='red', s=80, 
                          marker='*', label='烟雾弹起爆点')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.legend()
        ax1.set_title('3D场景图')
        
        # 俯视图
        ax2 = fig.add_subplot(222)
        ax2.scatter(self.fake_target[0], self.fake_target[1], color='red', s=100, label='假目标')
        ax2.scatter(self.real_target_center[0], self.real_target_center[1], color='green', s=100, label='真目标')
        ax2.scatter(self.missile_initial_pos[0], self.missile_initial_pos[1], color='blue', s=100, label='导弹初始位置')
        ax2.scatter(self.uav_initial_pos[0], self.uav_initial_pos[1], color='orange', s=100, label='无人机初始位置')
        
        if result['status'] != 'failed' and 'missile_speed' in result:
            # 投影轨迹到xy平面
            ax2.plot(missile_traj[:, 0], missile_traj[:, 1], 'b-', linewidth=2, label='导弹轨迹')
            ax2.plot(uav_positions[:, 0], uav_positions[:, 1], 'orange', linewidth=2, label='无人机轨迹')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.legend()
        ax2.set_title('俯视图 (XY平面)')
        ax2.grid(True)
        
        # 时间-高度图
        ax3 = fig.add_subplot(223)
        if result['status'] != 'failed' and 'missile_speed' in result:
            ax3.plot(t_points, missile_traj[:, 2], 'b-', linewidth=2, label='导弹高度')
            
            t_uav = np.linspace(0, uav_traj_time, len(uav_positions))
            ax3.plot(t_uav, uav_positions[:, 2], 'orange', linewidth=2, label='无人机高度')
            
            # 标记关键时间点
            ax3.axvline(x=result['drop_time'], color='purple', linestyle='--', label='投放时间')
            ax3.axvline(x=result['explosion_time'], color='red', linestyle='--', label='起爆时间')
        
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('高度 (m)')
        ax3.legend()
        ax3.set_title('高度-时间图')
        ax3.grid(True)
        
        # 遮蔽效果图
        ax4 = fig.add_subplot(224)
        if result['status'] != 'failed' and 'explosion_position' in result:
            # 计算遮蔽效果随时间的变化
            dt = 0.1
            times = []
            blocked_status = []
            missile_speed = result.get('missile_speed', 500)
            
            for t in np.arange(0, result['missile_flight_time'], dt):
                missile_pos = self.missile_position_at_time(t, missile_speed)
                smoke_pos = self.smoke_cloud_position(
                    result['explosion_position'], 
                    result['explosion_time'], 
                    t
                )
                
                times.append(t)
                if (smoke_pos is not None and 
                    smoke_pos[2] > 0 and 
                    t - result['explosion_time'] <= self.effective_duration):
                    blocked = self.is_missile_blocked(missile_pos, smoke_pos)
                    blocked_status.append(1 if blocked else 0)
                else:
                    blocked_status.append(0)
            
            ax4.fill_between(times, blocked_status, alpha=0.3, color='red', label='遮蔽状态')
            ax4.plot(times, blocked_status, 'r-', linewidth=2)
            ax4.axvline(x=result['explosion_time'], color='red', linestyle='--', alpha=0.7, label='起爆时间')
            
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('遮蔽状态')
        ax4.set_ylim(-0.1, 1.1)
        ax4.legend()
        ax4.set_title('遮蔽效果图')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存到: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    print("导弹干扰优化程序")
    print("="*50)
    
    # 创建优化器
    optimizer = MissileInterferenceOptimizer()
    
    # 尝试使用Gurobi求解（可能因为模型复杂度而失败）
    try:
        print("尝试使用Gurobi精确求解...")
        result = optimizer.solve_optimization()
        if result['status'] == 'failed':
            print("Gurobi求解失败，改用启发式算法...")
            result = optimizer.solve_simplified()
    except Exception as e:
        print(f"Gurobi求解出错: {e}")
        print("改用启发式算法...")
        result = optimizer.solve_simplified()
    
    # 打印结果
    optimizer.print_solution(result)
    
    # 可视化
    try:
        save_path = "results/figures/missile_interference_analysis.png"
        optimizer.visualize_scenario(result, save_path)
    except Exception as e:
        print(f"可视化出错: {e}")
        optimizer.visualize_scenario(result)


if __name__ == "__main__":
    main()
