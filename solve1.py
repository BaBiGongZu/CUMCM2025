"""
计算烟幕干扰弹对 M1 的有效遮蔽时长，并可视化爆炸瞬间。
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MissileInterferenceProblem1:
    def __init__(self):
        """初始化问题1的参数"""
        print("初始化问题1参数...")
        # 场景参数
        self.fake_target = np.array([0.0, 0.0, 0.0])
        self.real_target_center = np.array([0.0, 200.0, 0.0])
        self.real_target_radius = 7.0
        self.real_target_height = 10.0
        
        # 生成真目标圆柱体上的关键点（上下圆面各50个点）
        self.target_key_points = self._generate_target_key_points()
        
        # 导弹M1参数
        self.missile_pos = np.array([20000.0, 0.0, 2000.0])
        self.missile_speed = 300.0  # m/s
        self.missile_target = self.fake_target
        self.missile_distance = np.linalg.norm(self.missile_pos - self.missile_target)
        self.missile_direction = (self.missile_target - self.missile_pos) / self.missile_distance
        self.missile_flight_time = self.missile_distance / self.missile_speed

        # 无人机FY1参数
        self.uav_pos = np.array([17800.0, 0.0, 1800.0])
        self.uav_speed = 120.0  # m/s
        
        # 烟雾弹参数
        self.gravity = 9.8
        self.cloud_sink_speed = 3.0
        self.effective_radius = 10.0
        self.effective_duration = 20.0
        
        # 问题1特定时间参数
        self.t_drop_after_task = 1.5  # s
        self.t_explode_after_drop = 3.6  # s
        
        print(f"  - 导弹到假目标距离: {self.missile_distance:.2f} m")
        print(f"  - 导弹飞行总时间: {self.missile_flight_time:.2f} s")
        print(f"  - 真目标关键点数量: {len(self.target_key_points)}")

    def _generate_target_key_points(self, num_points_per_circle=50):
        """
        生成真目标圆柱体上的关键点
        在上下两个圆形截面上各均匀采样num_points_per_circle个点
        """
        key_points = []
        
        # 生成圆周上的角度
        angles = np.linspace(0, 2*np.pi, num_points_per_circle, endpoint=False)
        
        # 底面圆（z=0）
        for angle in angles:
            x = self.real_target_center[0] + self.real_target_radius * np.cos(angle)
            y = self.real_target_center[1] + self.real_target_radius * np.sin(angle)
            z = 0.0
            key_points.append([x, y, z])
        
        # 顶面圆（z=10）
        for angle in angles:
            x = self.real_target_center[0] + self.real_target_radius * np.cos(angle)
            y = self.real_target_center[1] + self.real_target_radius * np.sin(angle)
            z = self.real_target_height
            key_points.append([x, y, z])
        
        return np.array(key_points)

    def point_to_line_distance(self, point, line_start, line_end):
        """计算点到线段的距离"""
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

    def solve(self):
        """执行计算并返回包含所有结果的字典"""
        print("\n开始计算问题1...")

        # 1. 计算无人机飞行方向
        uav_dir_vec_xy = self.fake_target[:2] - self.uav_pos[:2]
        uav_direction = np.zeros(3)
        uav_direction[:2] = uav_dir_vec_xy / np.linalg.norm(uav_dir_vec_xy)
        
        # 2. 计算烟雾弹投放位置
        uav_drop_pos = self.uav_pos + self.uav_speed * self.t_drop_after_task * uav_direction
        
        # 3. 计算烟雾弹起爆位置
        bomb_initial_velocity = self.uav_speed * uav_direction
        explode_pos = uav_drop_pos.copy()
        explode_pos += bomb_initial_velocity * self.t_explode_after_drop
        explode_pos[2] -= 0.5 * self.gravity * self.t_explode_after_drop**2
        
        t_explode_abs = self.t_drop_after_task + self.t_explode_after_drop

        if explode_pos[2] < 0:
            print("  - 警告: 烟雾弹在落地后才起爆，可能无法形成有效遮蔽。")
            return {'total_blocked_time': 0.0}

        # 4. 模拟遮蔽过程
        print("开始模拟遮蔽过程（高精度采样）...")
        dt = 0.001  # 提高时间精度到0.001秒
        total_blocked_time = 0
        smoke_start_time = t_explode_abs
        smoke_end_time = smoke_start_time + self.effective_duration
        
        total_steps = int((min(smoke_end_time, self.missile_flight_time) - smoke_start_time) / dt)
        print(f"  - 时间步长: {dt:.3f} s")
        print(f"  - 总计算步数: {total_steps}")
        
        step_count = 0
        for t in np.arange(smoke_start_time, smoke_end_time, dt):
            if t > self.missile_flight_time: break
            
            current_missile_pos = self.missile_pos + self.missile_speed * t * self.missile_direction
            time_since_explode = t - smoke_start_time
            current_cloud_pos = explode_pos.copy()
            current_cloud_pos[2] -= self.cloud_sink_speed * time_since_explode
            if current_cloud_pos[2] < 0: break

            # 检查是否所有关键点都被遮蔽
            all_blocked = True
            for key_point in self.target_key_points:
                dist = self.point_to_line_distance(current_cloud_pos, current_missile_pos, key_point)
                if dist > self.effective_radius:
                    all_blocked = False
                    break
            
            if all_blocked:
                total_blocked_time += dt
            
            step_count += 1
            # 每1000步输出一次进度
            if step_count % 1000 == 0:
                progress = step_count / total_steps * 100
                print(f"  - 计算进度: {progress:.1f}%")
        
        print("模拟完成。")
        
        return {
            'total_blocked_time': total_blocked_time,
            'uav_direction': uav_direction,
            'uav_drop_pos': uav_drop_pos,
            'explode_pos': explode_pos,
            't_explode_abs': t_explode_abs,
        }

    def visualize_explosion_moment(self, results):
        """可视化烟雾弹爆炸瞬间的3D场景"""
        if 'explode_pos' not in results:
            print("无有效爆炸点，无法进行可视化。")
            return

        t_explode = results['t_explode_abs']
        explode_pos = results['explode_pos']
        
        # 爆炸瞬间的导弹位置
        missile_pos_at_explosion = self.missile_pos + self.missile_speed * t_explode * self.missile_direction

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 尝试设置中文字体，避免警告
        try:
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            # 如果设置失败，使用英文标签
            pass

        # 绘制假目标和真目标关键点
        ax.scatter(*self.fake_target, color='blue', s=100, marker='x', label='Fake Target')
        ax.scatter(self.target_key_points[:, 0], self.target_key_points[:, 1], self.target_key_points[:, 2], 
                   color='green', s=5, alpha=0.6, label='Target Key Points')

        # 绘制导弹轨迹和爆炸瞬间位置
        ax.plot([self.missile_pos[0], missile_pos_at_explosion[0]],
                [self.missile_pos[1], missile_pos_at_explosion[1]],
                [self.missile_pos[2], missile_pos_at_explosion[2]], 'r-', label='Missile Trajectory')
        ax.scatter(*missile_pos_at_explosion, color='red', s=100, marker='>', label='Missile at Explosion')

        # 绘制无人机轨迹和投放点
        ax.plot([self.uav_pos[0], results['uav_drop_pos'][0]],
                [self.uav_pos[1], results['uav_drop_pos'][1]],
                [self.uav_pos[2], results['uav_drop_pos'][2]], 'c-', label='UAV Trajectory')
        ax.scatter(*results['uav_drop_pos'], color='cyan', s=100, marker='v', label='Drop Point')

        # 绘制烟雾弹轨迹
        ax.plot([results['uav_drop_pos'][0], explode_pos[0]],
                [results['uav_drop_pos'][1], explode_pos[1]],
                [results['uav_drop_pos'][2], explode_pos[2]], 'k--', label='Smoke Bomb Trajectory')

        # 绘制烟雾云团
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = explode_pos[0] + self.effective_radius * np.outer(np.cos(u), np.sin(v))
        y = explode_pos[1] + self.effective_radius * np.outer(np.sin(u), np.sin(v))
        z = explode_pos[2] + self.effective_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.3)
        ax.scatter(*explode_pos, color='black', s=150, marker='*', label='Explosion Point')

        # 检查整体遮蔽效果
        blocked_count = 0
        for key_point in self.target_key_points:
            dist = self.point_to_line_distance(explode_pos, missile_pos_at_explosion, key_point)
            if dist <= self.effective_radius:
                blocked_count += 1
        
        total_points = len(self.target_key_points)
        blocking_ratio = blocked_count / total_points
        
        # 绘制少量代表性视线（避免图像过于复杂）
        sample_indices = np.linspace(0, len(self.target_key_points)-1, 8, dtype=int)
        for i, idx in enumerate(sample_indices):
            key_point = self.target_key_points[idx]
            dist = self.point_to_line_distance(explode_pos, missile_pos_at_explosion, key_point)
            is_blocked = dist <= self.effective_radius
            color = 'red' if is_blocked else 'green'
            alpha = 0.7 if is_blocked else 0.3
            ax.plot([missile_pos_at_explosion[0], key_point[0]],
                    [missile_pos_at_explosion[1], key_point[1]],
                    [missile_pos_at_explosion[2], key_point[2]],
                    color=color, linestyle='--', linewidth=1, alpha=alpha)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Smoke Interference at Explosion Moment\nBlocking: {blocked_count}/{total_points} points ({blocking_ratio:.1%})')
        
        # 调整图例
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()


def main():
    """主函数"""
    print("🎯 开始执行问题1求解程序")
    print("="*50)
    
    problem1 = MissileInterferenceProblem1()
    results = problem1.solve()
    
    print("\n" + "="*20 + " 计算结果 " + "="*20)
    print(f"✅ 对M1的有效遮蔽总时长为: {results.get('total_blocked_time', 0):.4f} 秒")
    print("="*50)

    # 可视化
    #problem1.visualize_explosion_moment(results)

if __name__ == "__main__":
    main()

