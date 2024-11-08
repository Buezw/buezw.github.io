from manim import *
import random
import time
import numpy as np

class CreateCircle(Scene):
    def construct(self):

        #random.seed(time.time())
        circle = Circle(radius=3)
        square = Square(side_length=6)
        side_length = MathTex("Side~Length = 6",font_size=30)
        side_length.next_to(square,RIGHT)
        radis_length = MathTex("Radis = 3",font_size=30)
        radis_length.move_to([1.5,0.5,0])
        radis_length.set_color(RED)
        radis = Line(start=[0,0,0],end=[3,0,0],color=RED)
        main = VGroup(circle,square,side_length,radis,radis_length)

        num_dots_one = 10
        num_dots_two = 100
        num_dots_three = 1000
        x_min, x_max = -3, 3
        y_min, y_max = -3, 3  # 用于追踪圆内点的数量
        numer = 0  # 计数器，记录圆内的点数
        denomi = 0 
        # 使用 Integer 显示数字，方便直接更新数值

        label = MathTex(r"\pi =")
        fraction = MathTex(r"\frac{0}{1} =")
        fraction.next_to(circle,LEFT,buff=1.5)
        number = DecimalNumber(0, num_decimal_places=5)
        group = VGroup(label,number).arrange(RIGHT)
        group.next_to(circle,RIGHT,buff=1)

        head = MathTex(r"Monte~Carlo~Approach~to~Calculate~\pi~-Buezwqwg")
        head.next_to(circle,UP,buff=0.5)
        head.set_color(BLUE)
        self.play(Create(main),Write(head),run_time=1)
        self.wait(2)
        self.play(Uncreate(head),Uncreate(side_length),Uncreate(radis_length),Uncreate(radis),run_time=0.1)
        self.play(main.animate.move_to(2.5*LEFT))
 
        area = MathTex(r"\frac{Area~of~Circle}{Area~of~Square}=\frac{\pi\cdot 3^2}{36}")
        pi_one = MathTex(r"\pi=\frac{A_C\cdot 36}{A_S\cdot 9}")
        pi_two = MathTex(r"\pi=\frac{A_C\cdot 4}{A_S}")
        pi_three = MathTex(r"\pi=\frac{A_C}{A_S}\cdot 4")
        area.move_to([2.5,0,0])
        pi_one.move_to([2.5,0,0])
        pi_two.move_to([2.5,0,0])
        pi_three.move_to([2.5,0,0])
        self.play(Write(area),run_time=0.5)
        self.wait(1)
        self.play(Transform(area,pi_one))
        self.wait(1)
        self.play(Transform(area,pi_two))
        self.wait(1)
        self.play(Transform(area,pi_three))
        self.wait(1)
        self.play(Uncreate(area),run_time=0.2)
        self.play(circle.animate.move_to(ORIGIN),square.animate.move_to(ORIGIN))

        self.play(Write(group),run_time=0.5)
        head = Text("When n=10",font_size=30)
        head.next_to(square,UP)
        self.play(Write(head))
        for _ in range(num_dots_one):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            dot = Dot(point=[x, y, 0], radius=0.01,color=YELLOW)
            self.play(FadeIn(dot), run_time=0.00001)
            distance = np.sqrt(x**2 + y**2)  # 计算点到圆心的距离
            if distance <= 3:
                numer+=1
                denomi+=1
                # 直接更新显示的数字
                new_fraction = MathTex(fr"\frac{{{numer}}}{{{denomi}}}")
                new_fraction.move_to(fraction.get_center())
                self.play(Transform(fraction, new_fraction),run_time=0.0001)
                self.play(number.animate.set_value((numer/denomi)*4),run_time=0.0001)
            else:
                denomi+=1
                self.play(number.animate.set_value((numer/denomi)*4),run_time=0.0001)
        self.wait(2)
        self.play(Uncreate(fraction),Uncreate(head))
        self.wait(2) 
        head_two = Text("Increase Number of Dots",font_size=30)
        head_two.next_to(square,UP,buff=0.5)
        self.play(Write(head_two),Uncreate(group),run_time=0.5)
        points = [Dot(point=[np.random.uniform(-3, 3), np.random.uniform(-3, 3), 0], radius=0.01, color=YELLOW) for _ in range(10000)]
        animations = [Create(dot) for dot in points]
        self.wait(2)
        self.play(Succession(*animations, lag_ratio=0.0001),Unwrite(head_two),)  # 控制动画间隔时间
        final = MathTex(r"\pi=\frac{7853}{10000}\cdot 4=3.1408859",font_size=30)
        final.next_to(circle,RIGHT)
        self.play(Write(final))
        self.wait(5)









