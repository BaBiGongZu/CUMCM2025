from gurobipy import *

def test_gurobi():
    """测试 Gurobi 是否能正常工作"""
    try:
        # 创建模型
        model = Model("test_model")

        # 添加变量
        x = model.addVar(name="x")
        y = model.addVar(name="y")

        # 设置目标函数: maximize x + 2*y
        model.setObjective(x + 2*y, GRB.MAXIMIZE)

        # 添加约束条件
        model.addConstr(x + y <= 3, "constraint1")
        model.addConstr(2*x + y <= 4, "constraint2")
        model.addConstr(x >= 0, "constraint3")
        model.addConstr(y >= 0, "constraint4")

        # 求解
        model.optimize()

        # 检查求解状态
        # if model.status == GRB.OPTIMAL:
            # print("✅ Gurobi 工作正常!")
            # print(f"最优解: x = {x.x:.2f}, y = {y.x:.2f}")
            # print(f"目标函数值: {model.objVal:.2f}")
        # else:
            # print("❌ 模型求解失败")

    except Exception as e:
        print(f"❌ Gurobi 测试失败: {e}")
        print("请检查 Gurobi 许可证或安装是否正确")

if __name__ == "__main__":
    test_gurobi()
