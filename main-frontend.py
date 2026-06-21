# main_customtkinter_enhanced.py (HulaDroneGUI with CustomTkinter - Enhanced)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.font_manager
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection registration
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import cv2
import threading
import time
import queue
import os
import json
from pathlib import Path, PosixPath
from typing import cast

from HulaDrone import HulaDrone # 无人机控制模块

class HulaDroneGUI_CTk_Enhanced:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Hula 无人机控制")
        self.root.geometry("800x400")  # 初始窗口大小较小，之后会调整
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing_window)

        # --- 主题设置 ---
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")

        # --- 加载字体 ---
        fonts_path = "./fonts/"
        if os.path.exists(fonts_path):
            font_files = [
                "PingFangSC-Light.otf", "PingFangSC-Medium.otf", 
                "PingFangSC-Regular.otf", "PingFangSC-Semibold.otf",
                "PingFangSC-Thin.otf", "PingFangSC-Ultralight.otf"
            ]
            for font in font_files:
                if os.path.exists(os.path.join(fonts_path, font)):
                    ctk.FontManager().load_font(os.path.join(fonts_path, font))

        # --- 样式设置 ---
        self.font_main = ("PingFangSC-Regular", 17)
        self.font_small = ("PingFangSC-Medium", 14)
        self.font_title = ("PingFangSC-Semibold", 22, "bold")
        self.corner_radius = 8
        self.padding = 10
        self.button_height = 35
        self.image_width = 640
        self.image_height = 360

        # --- 初始化无人机实例和状态 ---
        self.drone = HulaDrone()
        self.drone.register_status_callback(self.update_status_display_from_callback)

        self.default_ip = ""
        self.gui_active = False  # 跟踪GUI是否处于活动状态
        self.connected = False  # 跟踪连接状态
        
        # 飞行路径数据和视频流相关
        self.flight_path_data = {
            'x': [],
            'y': [],
            'z': [],
            'timestamps': []
        }
        self.target_path_data = {
            'x': [],
            'y': [],
            'z': []
        }
        self.site_map = self._load_site_map_prior()
        self.map_artists = []
        self.last_target_location = None
        self.drone_artists = []
        self.path_manual_view = False
        self.path_drag_state = None
        self.path_interaction_cids = []
        self.video_stream_active = False
        self.video_stream_show_target_frame = False # 是否在视频流中显示打靶目标框
        self.video_stream_show_distance_frame = False # 是否在视频流中显示单目测距结果
        self.laser_aim_target = False # 激光是否瞄准目标
        self.red_circle_laser_tracking = False
        self.image_raw_queue = queue.Queue(maxsize=1)  # 只保存最新图像帧
        self.image_update_queue = queue.Queue(maxsize=1)  # 只保存最新图像帧
        self.frame_queue = queue.Queue(maxsize=1)  # 只保存最新检测帧
        
        # --- 创建主容器 ---
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=self.padding, pady=self.padding)
        
        # --- 初始化只显示连接UI ---
        self.setup_connection_frame()
        
        # --- 创建但隐藏主界面框架 ---
        self.main_interface_frame = ctk.CTkFrame(self.root)
        # 不pack它，直到连接成功

        # # Track all scheduled callbacks and animations
        self.scheduled_callbacks = []
        self.cleanup_in_progress = False
        self._pending_status_update = None
        self._status_update_after_id = None
        self._status_update_interval_ms = 120
        self._last_path_ui_update = 0.0
        self._path_ui_interval_s = 0.33
        self._last_path_sample = None
        self._last_video_status_update = 0.0

        # self.show_main_interface()  # TODO：测试用
        
    def setup_connection_frame(self):
        """创建连接界面"""
        self.connection_frame = ctk.CTkFrame(self.main_container)
        self.connection_frame.pack(fill="both", expand=True)
        
        # 标题
        ctk.CTkLabel(
            self.connection_frame, 
            text="Hula 无人机控制系统", 
            font=self.font_title
        ).pack(pady=(20, 30))
        
        # IP输入框架
        ip_frame = ctk.CTkFrame(self.connection_frame, fg_color="transparent")
        ip_frame.pack(pady=10, fill="x", padx=50)
        
        ctk.CTkLabel(
            ip_frame, 
            text="无人机IP地址:", 
            font=self.font_main
        ).pack(side="left", padx=(0, 10))
        
        self.ip_entry = ctk.CTkEntry(
            ip_frame, 
            placeholder_text="输入IP地址或留空", 
            font=self.font_main,
            width=250,
            corner_radius=self.corner_radius
        )
        self.ip_entry.insert(0, self.default_ip)
        self.ip_entry.pack(side="left", fill="x", expand=True)
        
        # 连接按钮
        self.connect_button = ctk.CTkButton(
            self.connection_frame, 
            text="连接无人机", 
            command=self.action_connect_drone,
            font=self.font_main, 
            height=40, 
            width=200,
            corner_radius=self.corner_radius
        )
        self.connect_button.pack(pady=20)
        
        # 状态显示区域
        status_frame = ctk.CTkFrame(self.connection_frame, fg_color="transparent")
        status_frame.pack(pady=10, fill="x", padx=20)
        
        self.status_label = ctk.CTkLabel(
            status_frame, 
            text="状态: 等待连接", 
            font=self.font_main, 
            anchor="w"
        )
        self.status_label.pack(fill="x", pady=5)
        
        self.battery_label = ctk.CTkLabel(
            status_frame, 
            text="电池: --", 
            font=self.font_small, 
            anchor="w"
        )
        self.battery_label.pack(fill="x", pady=2)
        
        self.position_label = ctk.CTkLabel(
            status_frame, 
            text="位置: --", 
            font=self.font_small, 
            anchor="w"
        )
        self.position_label.pack(fill="x", pady=2)
        
        self.heading_label = ctk.CTkLabel(
            status_frame, 
            text="航向: --", 
            font=self.font_small, 
            anchor="w"
        )
        self.heading_label.pack(fill="x", pady=2)
        
    def setup_main_interface(self):
        """创建连接成功后的主界面"""
        # 如果已经创建过，就不重复创建
        if hasattr(self, 'main_interface_created') and self.main_interface_created:
            return
        
        # 设置标记，避免重复创建
        self.main_interface_created = True
        
        # 调整窗口大小以适应完整界面
        self.root.geometry("1200x900")
        
        # 主界面采用网格布局
        self.main_interface_frame.columnconfigure(0, weight=3)  # 控制区域
        self.main_interface_frame.columnconfigure(1, weight=2)  # 显示区域
        self.main_interface_frame.rowconfigure(0, weight=1)     # 单行布局
        
        # 创建左侧可滚动控制区域，避免窗口高度不足时底部功能被遮挡
        self.control_frame = ctk.CTkScrollableFrame(
            self.main_interface_frame,
            width=690,
            height=850,
            corner_radius=self.corner_radius,
        )
        self.control_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        
        # 配置控制区域网格
        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.rowconfigure(0, weight=0)  # 连接状态UI (已有信息复制过来)
        self.control_frame.rowconfigure(1, weight=0)  # 飞行控制UI
        self.control_frame.rowconfigure(2, weight=0)  # 功能控制UI
        self.control_frame.rowconfigure(3, weight=0)  # 自动飞行UI
        
        # 创建右侧显示区域
        self.display_frame = ctk.CTkFrame(self.main_interface_frame)
        self.display_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        
        # 配置显示区域网格
        self.display_frame.columnconfigure(0, weight=1)
        self.display_frame.rowconfigure(0, weight=1)  # 视频流
        self.display_frame.rowconfigure(1, weight=1)  # 飞行路径
        
        # --- 设置各子区域 ---
        
        # 1. 控制区域
        # 连接状态信息复制到新位置
        self.setup_connection_info_ui(self.control_frame)
        self.setup_laser_video_ui(self.control_frame)
        self.setup_flight_control_ui(self.control_frame)
        self.setup_auto_flight_ui(self.control_frame)
        self._bind_control_scrollwheel()
        
        # 2. 显示区域
        self.setup_display_ui()
        
    def show_main_interface(self):
        """显示主界面，隐藏连接界面"""
        # 确保已创建主界面
        self.setup_main_interface()
        
        # 隐藏连接界面
        self.connection_frame.pack_forget()
        self.main_container.pack_forget()
        
        # 显示主界面
        self.main_interface_frame.pack(fill="both", expand=True, padx=self.padding, pady=self.padding)
        
    def setup_display_ui(self):
        """设置显示区域：视频流和飞行路径"""
        # 视频流显示区域
        video_frame = self._create_section_frame(self.display_frame, "视频流显示", 0)
        video_frame.grid_rowconfigure(0, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)
        
        # 创建Matplotlib图形用于显示视频
        self.video_fig = plt.figure(figsize=(10.7, 3))
        self.video_ax = plt.Axes(self.video_fig, [0, 0, 1, 1], facecolor='black')
        self.video_ax.set_axis_off()
        self.video_ax.set_facecolor("black")
        self.video_fig.add_axes(self.video_ax)
        
        # 创建初始黑色图像
        self.video_img = self.video_ax.imshow(np.zeros((self.image_height, self.image_width, 3), dtype="uint8"))
        
        # 嵌入Matplotlib到Tkinter
        video_canvas = FigureCanvasTkAgg(self.video_fig, master=video_frame)
        self.video_canvas = video_canvas
        self.video_canvas_widget = video_canvas.get_tk_widget()
        self.video_canvas_widget.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.video_canvas_widget.config(background="black")
        
        # 添加状态文本显示
        self.video_status_label = ctk.CTkLabel(
            video_frame, 
            text="未开启视频流", 
            font=self.font_small,
            corner_radius=self.corner_radius
        )
        self.video_status_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        # 飞行路径显示区域
        path_frame = self._create_section_frame(self.display_frame, "飞行路径", 1)
        path_frame.grid_rowconfigure(0, weight=1)
        path_frame.grid_columnconfigure(0, weight=1)
        
        self._setup_rviz_path_scene()

        # 将Matplotlib图形嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(self.fig, master=path_frame)
        self.canvas = canvas
        canvas.draw()
        self.canvas_widget = canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self._bind_path_canvas_interactions()
        
        # 添加工具按钮
        tools_frame = ctk.CTkFrame(path_frame, fg_color="transparent")
        tools_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        ctk.CTkButton(
            tools_frame, 
            text="重置视图", 
            command=self.reset_path_view,
            height=30, 
            width=100,
            font=self.font_small, 
            corner_radius=self.corner_radius
        ).pack(side="left", padx=(0, 5))
        
        ctk.CTkButton(
            tools_frame, 
            text="清除轨迹", 
            command=self.clear_flight_path,
            height=30, 
            width=100,
            font=self.font_small, 
            corner_radius=self.corner_radius
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            tools_frame, 
            text="保存图像", 
            command=self.save_flight_path_image,
            height=30, 
            width=100,
            font=self.font_small, 
            corner_radius=self.corner_radius
        ).pack(side="left", padx=5)

    def _bind_path_canvas_interactions(self):
        if not hasattr(self, "canvas") or self.canvas is None:
            return
        self.path_interaction_cids = [
            self.canvas.mpl_connect("button_press_event", self._on_path_button_press),
            self.canvas.mpl_connect("button_release_event", self._on_path_button_release),
            self.canvas.mpl_connect("motion_notify_event", self._on_path_mouse_motion),
            self.canvas.mpl_connect("scroll_event", self._on_path_scroll),
        ]

    def _is_path_event(self, event):
        return (
            not self.cleanup_in_progress
            and hasattr(self, "ax")
            and event is not None
            and event.inaxes == self.ax
        )

    def _draw_path_canvas_idle(self):
        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            canvas.draw_idle()

    def _on_path_button_press(self, event):
        if not self._is_path_event(event):
            return
        if getattr(event, "dblclick", False):
            self.reset_path_view()
            return
        if event.button not in (1, 2, 3):
            return
        self.path_drag_state = {
            "button": event.button,
            "x": event.x,
            "y": event.y,
            "elev": self.ax.elev,
            "azim": self.ax.azim,
            "xlim": self.ax.get_xlim3d(),
            "ylim": self.ax.get_ylim3d(),
            "zlim": self.ax.get_zlim3d(),
        }

    def _on_path_button_release(self, event):
        self.path_drag_state = None

    def _on_path_mouse_motion(self, event):
        if not self._is_path_event(event) or not self.path_drag_state:
            return
        dx = event.x - self.path_drag_state["x"]
        dy = event.y - self.path_drag_state["y"]
        button = self.path_drag_state["button"]

        if button == 1:
            elev = max(-5, min(88, self.path_drag_state["elev"] - dy * 0.35))
            azim = self.path_drag_state["azim"] - dx * 0.35
            self.ax.view_init(elev=elev, azim=azim)
            self._draw_path_canvas_idle()
            return

        if button in (2, 3):
            width, height = self.fig.canvas.get_width_height()
            xlim = self.path_drag_state["xlim"]
            ylim = self.path_drag_state["ylim"]
            x_span = xlim[1] - xlim[0]
            y_span = ylim[1] - ylim[0]
            shift_x = -dx / max(width, 1) * x_span
            shift_y = dy / max(height, 1) * y_span
            self.ax.set_xlim3d(xlim[0] + shift_x, xlim[1] + shift_x)
            self.ax.set_ylim3d(ylim[0] + shift_y, ylim[1] + shift_y)
            self.path_manual_view = True
            self._draw_path_canvas_idle()

    def _on_path_scroll(self, event):
        if not self._is_path_event(event):
            return
        factor = 0.86 if event.step > 0 else 1.16
        xlim = self.ax.get_xlim3d()
        ylim = self.ax.get_ylim3d()
        zlim = self.ax.get_zlim3d()
        self._zoom_axis_3d(xlim, ylim, zlim, factor)
        self.path_manual_view = True
        self._draw_path_canvas_idle()

    def _zoom_axis_3d(self, xlim, ylim, zlim, factor):
        def scaled_limits(lim):
            center = (lim[0] + lim[1]) / 2
            half = (lim[1] - lim[0]) * factor / 2
            return center - half, center + half

        self.ax.set_xlim3d(*scaled_limits(xlim))
        self.ax.set_ylim3d(*scaled_limits(ylim))
        self.ax.set_zlim3d(*scaled_limits(zlim))

    def _get_path_font(self):
        font_path = Path("./fonts/PingFangSC-Regular.otf")
        if font_path.exists():
            return matplotlib.font_manager.FontProperties(fname=cast(PosixPath, font_path))
        return matplotlib.font_manager.FontProperties(family="Microsoft YaHei")

    def _load_site_map_prior(self):
        default_map = {
            "name": "fixed_training_field",
            "unit": "cm",
            "coordinate_frame": {
                "x_axis": "field_x_positive",
                "y_axis": "field_y_positive",
                "display_rotation_z_ccw_deg": -90,
            },
            "boundary": {
                "x_min": 0,
                "x_max": 682.1152,
                "y_min": 0,
                "y_max": 372.0,
                "z_min": 0,
                "z_max": 250,
            },
            "takeoff": {"x": 538.4037, "y": 143.7115, "z": 0},
            "floor_grid": {
                "tile_size_cm": 28.1923,
                "x_tiles": 24,
                "y_tiles": 13,
                "border_width_cm": 2.75,
                "style": "gray_white_checkerboard",
                "tile_alpha": 0.86,
            },
            "drone_visual": {
                "texture_path": "assets/drone_texture.png",
                "texture_size_cm": 82,
                "front_axis": "image_right",
                "max_texture_pixels": 768,
            },
            "rings": [],
            "pickup_points": [],
            "drop_points": [],
        }
        map_path = Path("site_map.json")
        if not map_path.exists():
            return default_map
        try:
            with map_path.open("r", encoding="utf-8") as f:
                loaded_map = json.load(f)
            return loaded_map
        except Exception as e:
            print(f"加载先验地图失败，使用默认地图: {e}")
            return default_map

    def _map_to_scene(self, x, y, z):
        rotation = self.site_map.get("coordinate_frame", {}).get("display_rotation_z_ccw_deg", 90)
        theta = np.deg2rad(float(rotation))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Rotate map points only for display; control/local coordinates stay unchanged.
        scene_x = float(x) * cos_t - float(y) * sin_t
        scene_y = float(x) * sin_t + float(y) * cos_t
        return scene_x, scene_y, float(z)

    def _map_many_to_scene(self, xs, ys, zs):
        scene_points = [self._map_to_scene(x, y, z) for x, y, z in zip(xs, ys, zs)]
        if not scene_points:
            return [], [], []
        scene_x, scene_y, scene_z = zip(*scene_points)
        return list(scene_x), list(scene_y), list(scene_z)

    def _scene_heading(self, heading):
        try:
            base_heading = float(heading)
        except (TypeError, ValueError):
            base_heading = 0.0
        rotation = self.site_map.get("coordinate_frame", {}).get("display_rotation_z_ccw_deg", 90)
        return 90.0 - base_heading + float(rotation)

    def _get_map_boundary(self):
        boundary = self.site_map.get("boundary", {})
        return {
            "x_min": float(boundary.get("x_min", 0)),
            "x_max": float(boundary.get("x_max", 682.1152)),
            "y_min": float(boundary.get("y_min", 0)),
            "y_max": float(boundary.get("y_max", 372.0)),
            "z_min": float(boundary.get("z_min", 0)),
            "z_max": float(boundary.get("z_max", 250)),
        }

    def _get_takeoff_point(self):
        takeoff = self.site_map.get("takeoff", {})
        return [
            float(takeoff.get("x", 538.4037)),
            float(takeoff.get("y", 143.7115)),
            float(takeoff.get("z", 0)),
        ]

    def _local_to_field(self, x, y, z):
        takeoff_x, takeoff_y, takeoff_z = self._get_takeoff_point()
        return takeoff_x + float(x), takeoff_y + float(y), takeoff_z + float(z)

    def _local_many_to_field(self, xs, ys, zs):
        field_points = [self._local_to_field(x, y, z) for x, y, z in zip(xs, ys, zs)]
        if not field_points:
            return [], [], []
        field_x, field_y, field_z = zip(*field_points)
        return list(field_x), list(field_y), list(field_z)

    def _local_many_to_scene(self, xs, ys, zs):
        field_x, field_y, field_z = self._local_many_to_field(xs, ys, zs)
        return self._map_many_to_scene(field_x, field_y, field_z)

    def _setup_rviz_path_scene(self):
        """Create an RViz-style real-time 3D flight scene."""
        self.path_font = self._get_path_font()
        self.fig = plt.figure(figsize=(5.6, 4.7), facecolor="#0b1118")
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="#0b1118")
        self.fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.93)

        self.ax.set_title("固定场地先验地图 + 实时三维飞行", fontproperties=self.path_font, color="#e5f4ff", pad=10)
        self.ax.set_xlabel("显示横轴 = 场地Y / cm", fontproperties=self.path_font, color="#58d68d", labelpad=8)
        self.ax.set_ylabel("显示纵轴 = -场地X / cm", fontproperties=self.path_font, color="#ff6b6b", labelpad=8)
        self.ax.set_zlabel("Z / cm", color="#5dade2", labelpad=8)
        self.ax.view_init(elev=35, azim=-90)
        self._style_rviz_axis()
        self._draw_rviz_reference_grid()
        self._draw_site_prior_map()

        self.ideal_line, = self.ax.plot([], [], [], linestyle="--", linewidth=2.6, color="#ffd166",
                                        alpha=0.95, label="理想轨迹")
        self.path_glow, = self.ax.plot([], [], [], linewidth=0.0, color="#0077ff",
                                       alpha=0.0, solid_capstyle="round")
        self.path_line, = self.ax.plot([], [], [], linewidth=2.0, color="#4b1f8f",
                                       alpha=0.98, solid_capstyle="round", label="实际轨迹")
        self.path_points = self.ax.scatter([], [], [], s=0, color="#f8ffff",
                                           edgecolors="#00d4ff", linewidths=0.0,
                                           depthshade=False, label="轨迹采样点")
        self.current_pos, = self.ax.plot([], [], [], marker="o", markersize=0, color="#00d8ff",
                                         markeredgecolor="#ffffff", markeredgewidth=0.0,
                                         linestyle="", label="当前位置")
        self.ground_pos = self.ax.scatter([], [], [], marker="o", s=0, color="#00d8ff",
                                          edgecolors="#ffffff", linewidths=0.0,
                                          alpha=0.0, depthshade=False)
        self.target_pos, = self.ax.plot([], [], [], marker="D", markersize=10, color="#ffd166",
                                        markeredgecolor="#fff6cc", markeredgewidth=1.0,
                                        linestyle="", label="当前目标")
        self.altitude_line, = self.ax.plot([], [], [], linestyle="-", linewidth=0.0,
                                           color="#00d8ff", alpha=0.0)

        takeoff_x, takeoff_y, takeoff_z = self._get_takeoff_point()
        self.path_hud = self.ax.text2D(
            0.03, 0.94,
            f"局部 X 0 Y 0 Z 0 cm  |  场地 X {takeoff_x:.1f} Y {takeoff_y:.1f}  点数 0",
            transform=self.ax.transAxes,
            color="#d9f6ff",
            fontproperties=self.path_font,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#101923", edgecolor="#25445c", alpha=0.9)
        )
        # Keep the live scene clean; HUD and colors are enough during flight.
        self._set_default_map_view()
        drone_x, drone_y, drone_z = self._map_to_scene(takeoff_x, takeoff_y, takeoff_z)
        self._draw_drone_model(drone_x, drone_y, drone_z, self._scene_heading(0))

    def _style_rviz_axis(self):
        self.ax.grid(True)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_tick_params(colors="#9fb6c8", labelsize=8)
            try:
                axis.pane.set_facecolor((0.04, 0.07, 0.10, 0.82))
                axis.pane.set_edgecolor("#243847")
                axis._axinfo["grid"]["color"] = (0.26, 0.42, 0.52, 0.38)
                axis._axinfo["grid"]["linewidth"] = 0.8
            except Exception:
                pass
        try:
            self.ax.w_xaxis.line.set_color("#3d5568")
            self.ax.w_yaxis.line.set_color("#3d5568")
            self.ax.w_zaxis.line.set_color("#3d5568")
        except Exception:
            pass

    def _draw_rviz_reference_grid(self):
        boundary = self._get_map_boundary()
        self._draw_checkerboard_floor(boundary)

        floor_grid = self.site_map.get("floor_grid", {})
        grid_step = float(floor_grid.get("tile_size_cm", 28.1923))
        border_width = max(0.0, float(floor_grid.get("border_width_cm", 0) or 0))
        x_tiles = int(floor_grid.get("x_tiles", 0) or 0)
        y_tiles = int(floor_grid.get("y_tiles", 0) or 0)
        grid_color = "#d2dae0"
        x_min, x_max = boundary["x_min"], boundary["x_max"]
        y_min, y_max = boundary["y_min"], boundary["y_max"]
        inner_x_min = x_min + border_width
        inner_y_min = y_min + border_width
        inner_x_max = min(x_max - border_width, inner_x_min + max(1, x_tiles) * grid_step)
        inner_y_max = min(y_max - border_width, inner_y_min + max(1, y_tiles) * grid_step)
        z_min = boundary["z_min"] + 0.3

        for x_index in range(max(1, x_tiles) + 1):
            x_value = min(inner_x_min + x_index * grid_step, inner_x_max)
            sx, sy, sz = self._map_many_to_scene(
                [x_value, x_value],
                [inner_y_min, inner_y_max],
                [z_min, z_min],
            )
            self.ax.plot(sx, sy, sz, color=grid_color, alpha=0.32, linewidth=0.38)

        for y_index in range(max(1, y_tiles) + 1):
            y_value = min(inner_y_min + y_index * grid_step, inner_y_max)
            sx, sy, sz = self._map_many_to_scene(
                [inner_x_min, inner_x_max],
                [y_value, y_value],
                [z_min, z_min],
            )
            self.ax.plot(sx, sy, sz, color=grid_color, alpha=0.32, linewidth=0.38)

        origin_x, origin_y, origin_z = self._map_to_scene(0, 0, 0)
        x_tip = self._map_to_scene(150, 0, 0)
        y_tip = self._map_to_scene(0, 150, 0)
        z_tip = self._map_to_scene(0, 0, 130)
        self.ax.quiver(origin_x, origin_y, origin_z, x_tip[0] - origin_x, x_tip[1] - origin_y, 0,
                       color="#ff5c5c", linewidth=2.0, arrow_length_ratio=0.12)
        self.ax.quiver(origin_x, origin_y, origin_z, y_tip[0] - origin_x, y_tip[1] - origin_y, 0,
                       color="#62d26f", linewidth=2.0, arrow_length_ratio=0.12)
        self.ax.quiver(origin_x, origin_y, origin_z, 0, 0, z_tip[2] - origin_z,
                       color="#58a6ff", linewidth=2.0, arrow_length_ratio=0.12)
        label_x = self._map_to_scene(170, 0, 0)
        label_y = self._map_to_scene(0, 170, 0)
        self.ax.text(label_x[0], label_x[1], label_x[2], "X正方向", color="#ff5c5c", fontsize=10, fontproperties=self.path_font)
        self.ax.text(label_y[0], label_y[1], label_y[2], "Y正方向", color="#62d26f", fontsize=10, fontproperties=self.path_font)
        self.ax.text(origin_x, origin_y, 145, "Z", color="#58a6ff", fontsize=10)

    def _draw_checkerboard_floor(self, boundary):
        floor_grid = self.site_map.get("floor_grid", {})
        tile_size = float(floor_grid.get("tile_size_cm", 28))
        if tile_size <= 0:
            tile_size = 28.0
        x_tiles = int(floor_grid.get("x_tiles", 0) or 0)
        y_tiles = int(floor_grid.get("y_tiles", 0) or 0)
        border_width = float(floor_grid.get("border_width_cm", 0) or 0)
        if x_tiles <= 0:
            x_tiles = max(1, int(round((boundary["x_max"] - boundary["x_min"]) / tile_size)))
        if y_tiles <= 0:
            y_tiles = max(1, int(round((boundary["y_max"] - boundary["y_min"]) / tile_size)))
        border_width = max(0.0, border_width)

        x_min, x_max = boundary["x_min"], boundary["x_max"]
        y_min, y_max = boundary["y_min"], boundary["y_max"]
        z_floor = boundary["z_min"]
        light_tile = "#ffffff"
        dark_tile = "#edf2f6"
        edge_color = "#ffffff"
        border_color = "#f8fafc"
        tile_alpha = float(floor_grid.get("tile_alpha", 0.86))
        tile_alpha = max(0.15, min(1.0, tile_alpha))

        def add_floor_quad(x0, x1, y0, y1, face_color, linewidth=0.18, alpha=None):
            if x1 <= x0 or y1 <= y0:
                return
            scene_x, scene_y, scene_z = self._map_many_to_scene(
                [x0, x1, x1, x0],
                [y0, y0, y1, y1],
                [z_floor, z_floor, z_floor, z_floor],
            )
            tile = Poly3DCollection(
                [list(zip(scene_x, scene_y, scene_z))],
                facecolors=face_color,
                edgecolors=edge_color,
                linewidths=linewidth,
                alpha=tile_alpha if alpha is None else alpha,
            )
            self.ax.add_collection3d(tile)

        inner_x_min = min(x_max, x_min + border_width)
        inner_y_min = min(y_max, y_min + border_width)
        inner_x_max = min(x_max - border_width, inner_x_min + x_tiles * tile_size)
        inner_y_max = min(y_max - border_width, inner_y_min + y_tiles * tile_size)

        add_floor_quad(x_min, x_max, y_min, inner_y_min, border_color, linewidth=0.12, alpha=tile_alpha)
        add_floor_quad(x_min, x_max, inner_y_max, y_max, border_color, linewidth=0.12, alpha=tile_alpha)
        add_floor_quad(x_min, inner_x_min, inner_y_min, inner_y_max, border_color, linewidth=0.12, alpha=tile_alpha)
        add_floor_quad(inner_x_max, x_max, inner_y_min, inner_y_max, border_color, linewidth=0.12, alpha=tile_alpha)

        for y_index in range(y_tiles):
            y0 = inner_y_min + y_index * tile_size
            y1 = min(y0 + tile_size, inner_y_max)
            for x_index in range(x_tiles):
                x0 = inner_x_min + x_index * tile_size
                x1 = min(x0 + tile_size, inner_x_max)
                face_color = light_tile if (x_index + y_index) % 2 == 0 else dark_tile
                add_floor_quad(x0, x1, y0, y1, face_color, linewidth=0.16)

    def _draw_site_prior_map(self):
        boundary = self._get_map_boundary()
        x_min, x_max = boundary["x_min"], boundary["x_max"]
        y_min, y_max = boundary["y_min"], boundary["y_max"]
        z_min, z_max = boundary["z_min"], boundary["z_max"]

        floor_corners = [
            (x_min, y_min, z_min),
            (x_max, y_min, z_min),
            (x_max, y_max, z_min),
            (x_min, y_max, z_min),
            (x_min, y_min, z_min),
        ]
        top_corners = [(x, y, z_max) for x, y, _ in floor_corners]
        floor_x, floor_y, floor_z = self._map_many_to_scene(
            [p[0] for p in floor_corners],
            [p[1] for p in floor_corners],
            [p[2] for p in floor_corners],
        )
        top_x, top_y, top_z = self._map_many_to_scene(
            [p[0] for p in top_corners],
            [p[1] for p in top_corners],
            [p[2] for p in top_corners],
        )
        self.ax.plot(floor_x, floor_y, floor_z, color="#8ecae6", linewidth=2.0, alpha=0.95, label="场地边界")
        self.ax.plot(top_x, top_y, top_z, color="#8ecae6", linewidth=1.2, alpha=0.42)

        for x_raw, y_raw, _ in floor_corners[:-1]:
            sx, sy, sz = self._map_many_to_scene([x_raw, x_raw], [y_raw, y_raw], [z_min, z_max])
            self.ax.plot(sx, sy, sz, color="#8ecae6", linewidth=1.0, alpha=0.35)

        takeoff_x, takeoff_y, takeoff_z = self._get_takeoff_point()
        scene_takeoff = self._map_to_scene(takeoff_x, takeoff_y, takeoff_z)
        self.ax.scatter([scene_takeoff[0]], [scene_takeoff[1]], [scene_takeoff[2]],
                        marker="*", s=0, color="#ffd166", edgecolors="#fff6cc",
                        linewidths=1.0, depthshade=False, label="起飞点")
        self.ax.text(scene_takeoff[0], scene_takeoff[1], scene_takeoff[2] + 18,
                     f"起飞点\nX{takeoff_x:.0f} Y{takeoff_y:.0f}", color="#ffd166", fontsize=8,
                     fontproperties=self.path_font, ha="center", alpha=0.0)

    def _set_3d_bounds(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_zlim(z_min, z_max)
        if hasattr(self.ax, "set_box_aspect"):
            self.ax.set_box_aspect((max(1, x_max - x_min), max(1, y_max - y_min), max(1, z_max - z_min) * 0.75))

    def _get_map_scene_bounds(self):
        boundary = self._get_map_boundary()
        corners = [
            (boundary["x_min"], boundary["y_min"], boundary["z_min"]),
            (boundary["x_max"], boundary["y_min"], boundary["z_min"]),
            (boundary["x_max"], boundary["y_max"], boundary["z_min"]),
            (boundary["x_min"], boundary["y_max"], boundary["z_min"]),
            (boundary["x_min"], boundary["y_min"], boundary["z_max"]),
            (boundary["x_max"], boundary["y_min"], boundary["z_max"]),
            (boundary["x_max"], boundary["y_max"], boundary["z_max"]),
            (boundary["x_min"], boundary["y_max"], boundary["z_max"]),
        ]
        scene_x, scene_y, scene_z = self._map_many_to_scene(
            [p[0] for p in corners],
            [p[1] for p in corners],
            [p[2] for p in corners],
        )
        return min(scene_x), max(scene_x), min(scene_y), max(scene_y), min(scene_z), max(scene_z)

    def _set_default_map_view(self):
        x_min, x_max, y_min, y_max, z_min, z_max = self._get_map_scene_bounds()
        margin_xy = 45
        margin_z = 25
        self._set_3d_bounds(x_min - margin_xy, x_max + margin_xy,
                            y_min - margin_xy, y_max + margin_xy,
                            max(0, z_min), z_max + margin_z)

    def _autoscale_3d_view(self):
        raw_xs = list(self.flight_path_data["x"]) + list(self.target_path_data["x"])
        raw_ys = list(self.flight_path_data["y"]) + list(self.target_path_data["y"])
        raw_zs = list(self.flight_path_data["z"]) + list(self.target_path_data["z"])
        if not raw_xs:
            self._set_default_map_view()
            return

        scene_xs, scene_ys, scene_zs = self._local_many_to_scene(raw_xs, raw_ys, raw_zs)
        xs = scene_xs
        ys = scene_ys
        zs = scene_zs + [0]

        margin_xy = 65
        margin_z = 65
        x_min, x_max = min(xs) - margin_xy, max(xs) + margin_xy
        y_min, y_max = min(ys) - margin_xy, max(ys) + margin_xy
        z_min, z_max = max(0, min(zs) - 25), max(zs) + margin_z
        span_xy = max(x_max - x_min, y_max - y_min, 300)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        self._set_3d_bounds(center_x - span_xy / 2, center_x + span_xy / 2,
                            center_y - span_xy / 2, center_y + span_xy / 2,
                            z_min, max(z_max, 130))

    def _get_current_target_location(self):
        controller = getattr(self.drone, "controller", None)
        if controller is None:
            return None
        try:
            target = controller.get_target_location()
        except Exception:
            return None
        if target and len(target) == 3:
            try:
                return [float(target[0]), float(target[1]), float(target[2])]
            except (TypeError, ValueError):
                return None
        return None

    def _update_target_path_from_controller(self):
        target = self._get_current_target_location()
        if target is None:
            self.target_pos.set_data_3d([], [], [])
            return

        should_append = self.last_target_location is None
        if self.last_target_location is not None:
            delta = sum(abs(target[i] - self.last_target_location[i]) for i in range(3))
            should_append = delta > 2

        if should_append:
            self.target_path_data["x"].append(target[0])
            self.target_path_data["y"].append(target[1])
            self.target_path_data["z"].append(target[2])
            self.last_target_location = target

        target_scene_x, target_scene_y, target_scene_z = self._local_many_to_scene(
            self.target_path_data["x"],
            self.target_path_data["y"],
            self.target_path_data["z"],
        )
        self.ideal_line.set_data_3d(target_scene_x, target_scene_y, target_scene_z)
        field_target = self._local_to_field(target[0], target[1], target[2])
        point_x, point_y, point_z = self._map_to_scene(field_target[0], field_target[1], field_target[2])
        self.target_pos.set_data_3d([point_x], [point_y], [point_z])

    def _load_drone_texture(self):
        if hasattr(self, "drone_texture_cache"):
            return self.drone_texture_cache

        visual = self.site_map.get("drone_visual", {})
        configured_path = visual.get("texture_path", "assets/drone_texture.png")
        candidates = [
            Path(configured_path),
            Path("assets/drone_texture.png"),
            Path("assets/drone.png"),
            Path("drone_texture.png"),
            Path("drone.png"),
        ]

        texture = None
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                texture = np.asarray(mpimg.imread(str(candidate)), dtype=float)
                self.drone_texture_path = str(candidate)
                break
            except Exception as e:
                print(f"加载无人机贴图失败 {candidate}: {e}")

        if texture is None:
            self.drone_texture_cache = None
            return None

        if texture.max() > 1.0:
            texture = texture / 255.0
        if texture.ndim == 2:
            texture = np.dstack([texture, texture, texture, np.ones_like(texture)])
        elif texture.shape[2] == 3:
            alpha = np.ones(texture.shape[:2], dtype=float)
            texture = np.dstack([texture, alpha])
        elif texture.shape[2] > 4:
            texture = texture[:, :, :4]

        # If the uploaded cutout is stored on a black background, make only near-black pixels transparent.
        rgb = texture[:, :, :3]
        alpha = texture[:, :, 3]
        if float(np.nanmin(alpha)) > 0.98:
            brightness = np.max(rgb, axis=2)
            keyed_alpha = np.clip((brightness - 0.025) / 0.10, 0.0, 1.0)
            texture[:, :, 3] = alpha * keyed_alpha

        visual = self.site_map.get("drone_visual", {})
        max_texture_pixels = int(visual.get("max_texture_pixels", 256))
        max_texture_pixels = max(96, min(1024, max_texture_pixels))
        max_pixels = max(texture.shape[0], texture.shape[1])
        if max_pixels > max_texture_pixels:
            scale = float(max_texture_pixels) / float(max_pixels)
            new_width = max(1, int(round(texture.shape[1] * scale)))
            new_height = max(1, int(round(texture.shape[0] * scale)))
            texture = cv2.resize(texture, (new_width, new_height), interpolation=cv2.INTER_AREA)
        texture = np.clip(texture, 0.0, 1.0)
        self.drone_texture_cache = texture
        return texture

    def _draw_drone_texture_model(self, x, y, z, heading=0):
        texture = self._load_drone_texture()
        if texture is None:
            return False

        try:
            heading_rad = np.deg2rad(float(heading))
        except (TypeError, ValueError):
            heading_rad = 0.0

        visual = self.site_map.get("drone_visual", {})
        size_cm = max(45.0, float(visual.get("texture_size_cm", 92)))
        image_front_axis = str(visual.get("front_axis", "image_right")).lower()
        tex_h, tex_w = texture.shape[:2]

        front_axis = np.array([np.cos(heading_rad), np.sin(heading_rad)])
        side_axis = np.array([-np.sin(heading_rad), np.cos(heading_rad)])

        if image_front_axis in ("image_right", "right", "x+"):
            half_front = size_cm / 2.0
            half_side = half_front * (float(tex_h) / max(1.0, float(tex_w)))
            front_coords = np.linspace(-half_front, half_front, tex_w)
            side_coords = np.linspace(half_side, -half_side, tex_h)
            front_grid, side_grid = np.meshgrid(front_coords, side_coords)
        elif image_front_axis in ("image_left", "left", "x-"):
            half_front = size_cm / 2.0
            half_side = half_front * (float(tex_h) / max(1.0, float(tex_w)))
            front_coords = np.linspace(half_front, -half_front, tex_w)
            side_coords = np.linspace(half_side, -half_side, tex_h)
            front_grid, side_grid = np.meshgrid(front_coords, side_coords)
        elif image_front_axis in ("image_bottom", "bottom", "y-"):
            half_front = size_cm / 2.0
            half_side = half_front * (float(tex_w) / max(1.0, float(tex_h)))
            front_coords = np.linspace(-half_front, half_front, tex_h)
            side_coords = np.linspace(-half_side, half_side, tex_w)
            side_grid, front_grid = np.meshgrid(side_coords, front_coords)
        else:
            half_front = size_cm / 2.0
            half_side = half_front * (float(tex_w) / max(1.0, float(tex_h)))
            front_coords = np.linspace(half_front, -half_front, tex_h)
            side_coords = np.linspace(-half_side, half_side, tex_w)
            side_grid, front_grid = np.meshgrid(side_coords, front_coords)

        x_grid = x + front_grid * front_axis[0] + side_grid * side_axis[0]
        y_grid = y + front_grid * front_axis[1] + side_grid * side_axis[1]
        z_grid = np.full_like(x_grid, z + 5.0)

        surface = self.ax.plot_surface(
            x_grid, y_grid, z_grid,
            facecolors=texture,
            linewidth=0,
            antialiased=True,
            shade=False,
            zorder=20,
        )
        self.drone_artists.append(surface)

        front_tip = np.array([x, y]) + front_axis * (half_front + 48)
        arrow_base = np.array([x, y]) + front_axis * (half_front * 0.16)
        heading_arrow_len = half_front * 0.78
        self.drone_artists.append(
            self.ax.quiver(
                arrow_base[0], arrow_base[1], z + 18,
                front_axis[0] * heading_arrow_len, front_axis[1] * heading_arrow_len, 0,
                color="#ffd166", linewidth=2.4, arrow_length_ratio=0.24
            )
        )
        self.drone_artists.append(
            self.ax.quiver(
                x, y, 2,
                front_axis[0] * heading_arrow_len, front_axis[1] * heading_arrow_len, 0,
                color="#ffd166", linewidth=0.0, alpha=0.0, arrow_length_ratio=0.22
            )
        )
        self.drone_artists.append(
            self.ax.text(front_tip[0], front_tip[1], z + 34, "机头朝向",
                         color="#ffd166", fontsize=0, alpha=0.0,
                         fontproperties=self.path_font, ha="center",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="#101923",
                                   edgecolor="#ffd166", alpha=0.0))
        )

        theta = np.linspace(0, 2 * np.pi, 48)
        shadow_radius = size_cm * 0.34
        shadow_x = x + shadow_radius * np.cos(theta)
        shadow_y = y + shadow_radius * np.sin(theta)
        self.drone_artists.append(
            self.ax.plot(shadow_x, shadow_y, np.zeros_like(theta),
                         color="#071018", linewidth=3.0, alpha=0.26)[0]
        )
        self.drone_artists.append(
            self.ax.scatter([x], [y], [z + 7], s=0, color="#ffffff",
                            edgecolors="#00e5ff", linewidths=0.0, alpha=0.0, depthshade=False)
        )
        return True

    def _draw_drone_model(self, x, y, z, heading=0):
        for artist in self.drone_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.drone_artists = []

        if self._draw_drone_texture_model(x, y, z, heading):
            return

        try:
            heading_rad = np.deg2rad(float(heading))
        except (TypeError, ValueError):
            heading_rad = 0.0

        cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        arm = 42
        rotor_radius = 13
        local_points = {
            "front": np.array([arm, 0.0]),
            "back": np.array([-arm, 0.0]),
            "left": np.array([0.0, arm]),
            "right": np.array([0.0, -arm]),
        }
        world = {name: rot.dot(point) + np.array([x, y]) for name, point in local_points.items()}

        arm_color = "#ffffff"
        accent = "#00e5ff"
        front_color = "#ffd166"
        self.drone_artists.append(self.ax.plot([world["front"][0], world["back"][0]],
                                               [world["front"][1], world["back"][1]],
                                               [z, z], color=arm_color, linewidth=5.0,
                                               solid_capstyle="round")[0])
        self.drone_artists.append(self.ax.plot([world["left"][0], world["right"][0]],
                                               [world["left"][1], world["right"][1]],
                                               [z, z], color=arm_color, linewidth=5.0,
                                               solid_capstyle="round")[0])
        self.drone_artists.append(self.ax.scatter([x], [y], [z], s=260, color="#ffffff",
                                                  edgecolors=accent, linewidths=2.4, depthshade=False))
        fallback_axis = np.array([cos_h, sin_h])
        fallback_arrow_len = 85
        self.drone_artists.append(self.ax.quiver(x, y, z + 14, cos_h * fallback_arrow_len, sin_h * fallback_arrow_len, 0,
                                                 color=front_color, linewidth=4.4, arrow_length_ratio=0.24))
        self.drone_artists.append(self.ax.quiver(x, y, 2, cos_h * fallback_arrow_len, sin_h * fallback_arrow_len, 0,
                                                 color=front_color, linewidth=2.4, alpha=0.68, arrow_length_ratio=0.24))
        fallback_front_tip = np.array([x, y]) + fallback_axis * (fallback_arrow_len + 18)
        self.drone_artists.append(
            self.ax.text(fallback_front_tip[0], fallback_front_tip[1], z + 34, "机头朝向",
                         color=front_color, fontsize=11,
                         fontproperties=self.path_font, ha="center",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="#101923",
                                   edgecolor=front_color, alpha=0.78))
        )

        theta = np.linspace(0, 2 * np.pi, 34)
        for name, point in world.items():
            color = front_color if name == "front" else accent
            cx, cy = point[0], point[1]
            circle_x = cx + rotor_radius * np.cos(theta)
            circle_y = cy + rotor_radius * np.sin(theta)
            circle_z = np.full_like(theta, z)
            self.drone_artists.append(self.ax.plot(circle_x, circle_y, circle_z,
                                                   color=color, linewidth=2.8, alpha=0.98)[0])
            self.drone_artists.append(self.ax.scatter([cx], [cy], [z], s=20,
                                                      color=color, depthshade=False))

        self.drone_artists.append(
            self.ax.text(x, y, z + 28, "DRONE", color="#ffffff", fontsize=9,
                         fontproperties=self.path_font, ha="center")
        )

        shadow_radius = 28
        shadow_x = x + shadow_radius * np.cos(theta)
        shadow_y = y + shadow_radius * np.sin(theta)
        self.drone_artists.append(self.ax.plot(shadow_x, shadow_y, np.zeros_like(theta),
                                               color="#ff2bd6", linewidth=2.2, alpha=0.55)[0])

    def _create_section_frame(self, parent, title_text, row, column=0, columnspan=1):
        """创建带标题的区域框架"""
        section_container = ctk.CTkFrame(parent, fg_color="transparent")
        section_container.grid(row=row, column=column, columnspan=columnspan, sticky="ew", padx=self.padding, pady=self.padding)
        
        section_container.columnconfigure(0, weight=1)
        section_container.rowconfigure(0, weight=0)  # 标题
        section_container.rowconfigure(1, weight=0)  # 内容
        
        # 标题
        section_title = ctk.CTkLabel(section_container, text=title_text, font=self.font_title, anchor="w")
        section_title.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        # 内容框架
        frame = ctk.CTkFrame(section_container, corner_radius=self.corner_radius)
        frame.grid(row=1, column=0, sticky="ew")
        
        return frame

    def _bind_control_scrollwheel(self):
        """Bind mouse wheel events to the scrollable control panel."""
        if not hasattr(self.control_frame, "_parent_canvas"):
            return

        canvas = self.control_frame._parent_canvas

        def on_mousewheel(event):
            canvas.yview_scroll(-int(event.delta / 120), "units")

        def bind_recursive(widget):
            widget.bind("<MouseWheel>", on_mousewheel, add="+")
            for child in widget.winfo_children():
                bind_recursive(child)

        bind_recursive(self.control_frame)

    ## --- 连接与状态 UI ---
    def setup_connection_info_ui(self, parent_container):
        frame = self._create_section_frame(parent_container, "连接与状态", 0)
        frame.grid_columnconfigure(1, weight=1) # IP entry expands
        
        # 状态区域
        status_sub_frame = ctk.CTkFrame(frame, fg_color="transparent")
        status_sub_frame.grid(row=1, column=0, columnspan=3, padx=self.padding, pady=self.padding, sticky="ew")
        status_sub_frame.grid_columnconfigure((0,1,2), weight=1)

        # 复制状态标签
        self.main_status_label = ctk.CTkLabel(status_sub_frame, text="状态: 已连接", font=self.font_main, anchor="w")
        self.main_status_label.grid(row=0, column=0, columnspan=3, pady=(5,10), sticky="ew")

        self.main_battery_label = ctk.CTkLabel(status_sub_frame, text=self.battery_label.cget("text"), font=self.font_small, anchor="w")
        self.main_battery_label.grid(row=1, column=0, pady=2, sticky="w")

        self.main_position_label = ctk.CTkLabel(status_sub_frame, text=self.position_label.cget("text"), font=self.font_small, anchor="w")
        self.main_position_label.grid(row=1, column=1, pady=2, sticky="w")

        self.main_heading_label = ctk.CTkLabel(status_sub_frame, text=self.heading_label.cget("text"), font=self.font_small, anchor="w")
        self.main_heading_label.grid(row=1, column=2, pady=2, sticky="w")

    ## --- 飞行控制 UI ---
    def setup_flight_control_ui(self, parent_container):
        frame = self._create_section_frame(parent_container, "飞行控制", 2)
        frame.grid_columnconfigure((1,3,5), weight=0) # Entries fixed width
        frame.grid_columnconfigure(6, weight=1) # Move button can expand

        # Row 0: Core Actions
        core_actions_frame = ctk.CTkFrame(frame, fg_color="transparent")
        core_actions_frame.grid(row=0, column=0, columnspan=7, pady=self.padding, padx=self.padding, sticky="ew")
        core_actions_frame.grid_columnconfigure((0,1,2), weight=1) # Distribute buttons evenly

        ctk.CTkButton(core_actions_frame, text="准备", command=self.action_start_services, height=self.button_height, font=self.font_main, corner_radius=self.corner_radius).grid(row=0, column=0, padx=5, sticky="ew")
        ctk.CTkButton(core_actions_frame, text="起飞", command=self.action_takeoff, height=self.button_height, font=self.font_main, corner_radius=self.corner_radius).grid(row=0, column=1, padx=5, sticky="ew")
        ctk.CTkButton(core_actions_frame, text="降落", command=self.action_land, height=self.button_height, font=self.font_main, corner_radius=self.corner_radius).grid(row=0, column=2, padx=5, sticky="ew")

        # Row 1: Movement
        move_entry_width = 80
        ctk.CTkLabel(frame, text="X (cm):", font=self.font_main).grid(row=1, column=0, padx=(self.padding,0), pady=self.padding, sticky="e")
        self.x_entry = ctk.CTkEntry(frame, width=move_entry_width, font=self.font_main, corner_radius=self.corner_radius); self.x_entry.insert(0, "0")
        self.x_entry.grid(row=1, column=1, padx=5, pady=self.padding)

        ctk.CTkLabel(frame, text="Y (cm):", font=self.font_main).grid(row=1, column=2, padx=(self.padding,0), pady=self.padding, sticky="e")
        self.y_entry = ctk.CTkEntry(frame, width=move_entry_width, font=self.font_main, corner_radius=self.corner_radius); self.y_entry.insert(0, "0")
        self.y_entry.grid(row=1, column=3, padx=5, pady=self.padding)

        ctk.CTkLabel(frame, text="Z (cm):", font=self.font_main).grid(row=1, column=4, padx=(self.padding,0), pady=self.padding, sticky="e")
        self.z_entry = ctk.CTkEntry(frame, width=move_entry_width, font=self.font_main, corner_radius=self.corner_radius); self.z_entry.insert(0, "50")
        self.z_entry.grid(row=1, column=5, padx=5, pady=self.padding)

        ctk.CTkButton(frame, text="移动", command=self.action_move_to_target, height=self.button_height, font=self.font_main, corner_radius=self.corner_radius).grid(row=1, column=6, padx=(self.padding, self.padding), pady=self.padding, sticky="ew")

        # Row 2: Rotation
        ctk.CTkLabel(frame, text="右转 (度):", font=self.font_main).grid(row=2, column=0, padx=(self.padding,0), pady=self.padding, sticky="e")
        self.rotation_entry = ctk.CTkEntry(frame, width=move_entry_width, font=self.font_main, corner_radius=self.corner_radius); self.rotation_entry.insert(0, "0")
        self.rotation_entry.grid(row=2, column=1, padx=5, pady=self.padding)
        ctk.CTkButton(frame, text="开始旋转", command=self.action_rotation, height=self.button_height, font=self.font_main, corner_radius=self.corner_radius).grid(row=2, column=2, columnspan=2, padx=(self.padding, 5), pady=self.padding, sticky="ew")

        # Row 2: Set Heading
        ctk.CTkLabel(frame, text="航向 (度):", font=self.font_main).grid(row=2, column=4, padx=(self.padding,0), pady=self.padding, sticky="e")
        self.heading_entry = ctk.CTkEntry(frame, width=move_entry_width, font=self.font_main, corner_radius=self.corner_radius); self.heading_entry.insert(0, "0")
        self.heading_entry.grid(row=2, column=5, padx=5, pady=self.padding)
        ctk.CTkButton(
            frame, text="设置航向", command=self.action_set_heading,
            height=self.button_height, font=self.font_main, corner_radius=self.corner_radius
        ).grid(row=2, column=6, padx=(self.padding, 5), pady=self.padding, sticky="ew")

        # Row 3: Manual Control
        # 添加方向控制界面
        self.setup_direction_control_ui(frame, 3)  # 从第3行开始

        # Row 4: Detect and aim routine
        ctk.CTkButton(
            frame, text="目标检测", command=self.action_detect,
            height=self.button_height, font=self.font_main, corner_radius=self.corner_radius
        ).grid(row=4, column=0, columnspan=2, padx=(self.padding, 5), pady=self.padding, sticky="ew")
        ctk.CTkButton(
            frame, text="瞄准", command=self.action_toggle_aim_target,
            height=self.button_height, font=self.font_main, corner_radius=self.corner_radius
        ).grid(row=4, column=2, columnspan=2, padx=(self.padding, 5), pady=self.padding, sticky="ew")

    def setup_direction_control_ui(self, parent_container, start_row):
        """添加更紧凑的方向控制按钮界面"""
        # # 创建标题标签
        # ctk.CTkLabel(
        #     parent_container, 
        #     text="方向控制", 
        #     font=self.font_title, 
        #     anchor="w"
        # ).grid(row=start_row, column=0, columnspan=7, sticky="w", pady=(15, 5), padx=self.padding)
        
        # 创建主控制框架
        control_frame = ctk.CTkFrame(parent_container, fg_color="transparent")
        control_frame.grid(row=start_row, column=0, columnspan=7, sticky="ew", padx=self.padding, pady=(0, 5))
        
        # 将控制框架分为左右两部分
        control_frame.grid_columnconfigure(0, weight=1)  # 左侧 - 相机+航向控制
        control_frame.grid_columnconfigure(1, weight=1)  # 右侧 - 移动控制
        
        # ===== 左侧：相机+航向控制 =====
        cam_heading_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        cam_heading_frame.grid(row=0, column=0, sticky="nsew", padx=0)
        cam_heading_frame.grid_columnconfigure((0, 1), weight=1)
        
        # --- 左上：相机控制 ---
        cam_control_frame = ctk.CTkFrame(cam_heading_frame, fg_color=("gray90", "gray20"))
        cam_control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # 相机控制标题和步长
        ctk.CTkLabel(
            cam_control_frame, 
            text="相机俯仰控制", 
            font=self.font_small, 
            anchor="center"
        ).pack(pady=(5, 2))
        
        step_frame = ctk.CTkFrame(cam_control_frame, fg_color="transparent")
        step_frame.pack(fill="x", pady=2)
        
        ctk.CTkLabel(
            step_frame, 
            text="步长:", 
            font=("PingFangSC-Regular", 12)
        ).pack(side="left", padx=(45, 5))
        
        self.camera_step_entry = ctk.CTkEntry(
            step_frame, 
            width=40, 
            height=25,
            font=("PingFangSC-Regular", 12), 
            corner_radius=self.corner_radius
        )
        self.camera_step_entry.pack(side="left")
        self.camera_step_entry.insert(0, "5")
        
        ctk.CTkLabel(step_frame, text="°", font=("PingFangSC-Regular", 12)).pack(side="left")
        
        # 相机控制按钮
        buttons_frame = ctk.CTkFrame(cam_control_frame, fg_color="transparent")
        buttons_frame.pack(fill="both", expand=True, pady=3)
        
        # 上按钮
        ctk.CTkButton(
            buttons_frame, 
            text="↑", 
            command=self.action_camera_pitch_up,
            width=45, 
            height=30, 
            font=(self.font_main[0], 18), 
            corner_radius=self.corner_radius
        ).grid(row=0, column=0, pady=(0, 5), padx=(60, 0))
        
        # 下按钮
        ctk.CTkButton(
            buttons_frame, 
            text="↓", 
            command=self.action_camera_pitch_down,
            width=45, 
            height=30, 
            font=(self.font_main[0], 18), 
            corner_radius=self.corner_radius
        ).grid(row=1, column=0, pady=(0, 25), padx=(60, 0))
        
        # --- 右上：航向控制 ---
        heading_control_frame = ctk.CTkFrame(cam_heading_frame, fg_color=("gray90", "gray20"))
        heading_control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # 航向控制标题和步长
        ctk.CTkLabel(
            heading_control_frame, 
            text="航向控制", 
            font=self.font_small, 
            anchor="center"
        ).pack(pady=(5, 2))
        
        step_frame = ctk.CTkFrame(heading_control_frame, fg_color="transparent")
        step_frame.pack(fill="x", pady=2)
        
        ctk.CTkLabel(
            step_frame, 
            text="步长:", 
            font=("PingFangSC-Regular", 12)
        ).pack(side="left", padx=(50, 5))
        
        self.heading_step_entry = ctk.CTkEntry(
            step_frame, 
            width=40,
            height=25,
            font=("PingFangSC-Regular", 12), 
            corner_radius=self.corner_radius
        )
        self.heading_step_entry.pack(side="left")
        self.heading_step_entry.insert(0, "5")
        
        ctk.CTkLabel(step_frame, text="°", font=("PingFangSC-Regular", 12)).pack(side="left")
        
        # 航向控制按钮
        buttons_frame = ctk.CTkFrame(heading_control_frame, fg_color="transparent")
        buttons_frame.pack(fill="both", expand=True, pady=3)
        
        # 左按钮
        ctk.CTkButton(
            buttons_frame, 
            text="←", 
            command=self.action_heading_left,
            width=45, 
            height=30, 
            font=(self.font_main[0], 18), 
            corner_radius=self.corner_radius
        ).grid(row=0, column=0, padx=(35, 5))
        
        # 右按钮
        ctk.CTkButton(
            buttons_frame, 
            text="→", 
            command=self.action_heading_right,
            width=45, 
            height=30, 
            font=(self.font_main[0], 18), 
            corner_radius=self.corner_radius
        ).grid(row=0, column=1)
        
        # ===== 右侧：移动控制 =====
        movement_frame = ctk.CTkFrame(control_frame, fg_color=("gray90", "gray20"))
        movement_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # 移动控制标题和步长
        ctk.CTkLabel(
            movement_frame, 
            text="无人机移动控制", 
            font=self.font_small, 
            anchor="center"
        ).pack(pady=(5, 2))
        
        step_frame = ctk.CTkFrame(movement_frame, fg_color="transparent")
        step_frame.pack(fill="x", pady=2)
        
        ctk.CTkLabel(
            step_frame, 
            text="步长:", 
            font=("PingFangSC-Regular", 12)
        ).pack(side="left", padx=(5, 2))
        
        self.movement_step_entry = ctk.CTkEntry(
            step_frame, 
            width=40,
            height=25,
            font=("PingFangSC-Regular", 12), 
            corner_radius=self.corner_radius
        )
        self.movement_step_entry.pack(side="left")
        self.movement_step_entry.insert(0, "10")
        
        ctk.CTkLabel(step_frame, text="cm", font=("PingFangSC-Regular", 12)).pack(side="left")
        
        # 移动控制按钮
        movement_buttons_frame = ctk.CTkFrame(movement_frame, fg_color="transparent")
        movement_buttons_frame.pack(fill="both", expand=True, pady=(3, 5))
        movement_buttons_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        button_size = 35
        button_font = (self.font_small[0], 14)
        
        # 布局六个移动按钮
        # 前后上下按钮
        ctk.CTkButton(
            movement_buttons_frame, 
            text="前", 
            command=self.action_move_forward,
            width=button_size, 
            height=button_size, 
            font=button_font, 
            corner_radius=self.corner_radius
        ).grid(row=0, column=1, padx=2, pady=2)
        
        ctk.CTkButton(
            movement_buttons_frame, 
            text="后", 
            command=self.action_move_backward,
            width=button_size, 
            height=button_size, 
            font=button_font, 
            corner_radius=self.corner_radius
        ).grid(row=1, column=1, padx=2, pady=2)
        
        ctk.CTkButton(
            movement_buttons_frame, 
            text="上", 
            command=self.action_move_up,
            width=button_size, 
            height=button_size, 
            font=button_font, 
            corner_radius=self.corner_radius
        ).grid(row=0, column=3, padx=2, pady=2)
        
        ctk.CTkButton(
            movement_buttons_frame, 
            text="下", 
            command=self.action_move_down,
            width=button_size, 
            height=button_size, 
            font=button_font, 
            corner_radius=self.corner_radius
        ).grid(row=1, column=3, padx=2, pady=2)
        
        # 左右按钮
        ctk.CTkButton(
            movement_buttons_frame, 
            text="左", 
            command=self.action_move_left,
            width=button_size, 
            height=button_size, 
            font=button_font, 
            corner_radius=self.corner_radius
        ).grid(row=1, column=0, padx=2, pady=2)
        
        ctk.CTkButton(
            movement_buttons_frame, 
            text="右", 
            command=self.action_move_right,
            width=button_size, 
            height=button_size, 
            font=button_font, 
            corner_radius=self.corner_radius
        ).grid(row=1, column=2, padx=2, pady=2)

    ## --- 激光与视频流 UI ---
    def setup_laser_video_ui(self, parent_container):
        frame = self._create_section_frame(parent_container, "功能控制", 1)
        frame.grid_columnconfigure(1, weight=1) # Stream button expands

        self.laser_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            frame, text="发射激光", variable=self.laser_var, command=self.action_toggle_laser,
            onvalue=True, offvalue=False, font=self.font_main, corner_radius=self.corner_radius,
            border_width=2
        ).grid(row=0, column=0, padx=self.padding, pady=self.padding)

        self.video_stream_button = ctk.CTkButton(
            frame, text="开启视频流", command=self.action_capture_image_stream,
            height=self.button_height, font=self.font_main, corner_radius=self.corner_radius
        )
        self.video_stream_button.grid(row=0, column=1, padx=(0, self.padding), pady=self.padding, sticky="ew")

        self.monocular_distance_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            frame, text="单目测距", variable=self.monocular_distance_var,
            command=self.action_toggle_monocular_distance,
            onvalue=True, offvalue=False, font=self.font_main,
            corner_radius=self.corner_radius, border_width=2
        ).grid(row=1, column=0, padx=self.padding, pady=(0, self.padding), sticky="w")

        distance_config_frame = ctk.CTkFrame(frame, fg_color="transparent")
        distance_config_frame.grid(row=1, column=1, padx=(0, self.padding), pady=(0, self.padding), sticky="ew")
        distance_config_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(distance_config_frame, text="红色圆片直径(cm)", font=self.font_small).grid(row=0, column=0, padx=(0, 5), sticky="e")
        self.distance_diameter_entry = ctk.CTkEntry(
            distance_config_frame, width=70, font=self.font_small,
            corner_radius=self.corner_radius
        )
        self.distance_diameter_entry.insert(0, "10")
        self.distance_diameter_entry.grid(row=0, column=1, padx=(0, 10), sticky="ew")

        ctk.CTkButton(
            distance_config_frame, text="应用", command=self.action_apply_monocular_reference,
            height=28, width=60, font=self.font_small, corner_radius=self.corner_radius
        ).grid(row=0, column=2, sticky="e")

        self.red_circle_track_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            frame, text="红圆激光跟踪", variable=self.red_circle_track_var,
            command=self.action_toggle_red_circle_laser_tracking,
            onvalue=True, offvalue=False, font=self.font_main,
            corner_radius=self.corner_radius, border_width=2
        ).grid(row=2, column=0, padx=self.padding, pady=(0, self.padding), sticky="w")

        self.red_circle_track_label = ctk.CTkLabel(
            frame,
            text="激光点自动对准红圆圆心",
            font=self.font_small,
            anchor="w"
        )
        self.red_circle_track_label.grid(row=2, column=1, padx=(0, self.padding), pady=(0, self.padding), sticky="ew")

        offset_frame = ctk.CTkFrame(frame, fg_color="transparent")
        offset_frame.grid(row=3, column=0, columnspan=2, padx=self.padding, pady=(0, self.padding), sticky="ew")
        offset_frame.grid_columnconfigure((1, 3), weight=1)

        ctk.CTkLabel(offset_frame, text="激光偏移X(px)", font=self.font_small).grid(row=0, column=0, padx=(0, 5), sticky="e")
        self.laser_offset_x_entry = ctk.CTkEntry(
            offset_frame, width=70, font=self.font_small,
            corner_radius=self.corner_radius
        )
        self.laser_offset_x_entry.insert(0, "0")
        self.laser_offset_x_entry.grid(row=0, column=1, padx=(0, 10), sticky="ew")

        ctk.CTkLabel(offset_frame, text="Y(px)", font=self.font_small).grid(row=0, column=2, padx=(0, 5), sticky="e")
        self.laser_offset_y_entry = ctk.CTkEntry(
            offset_frame, width=70, font=self.font_small,
            corner_radius=self.corner_radius
        )
        self.laser_offset_y_entry.insert(0, "0")
        self.laser_offset_y_entry.grid(row=0, column=3, padx=(0, 10), sticky="ew")

        ctk.CTkButton(
            offset_frame, text="应用偏移", command=self.action_apply_red_circle_laser_offset,
            height=28, width=80, font=self.font_small, corner_radius=self.corner_radius
        ).grid(row=0, column=4, sticky="e")

    ## --- 自动飞行 UI ---
    def setup_auto_flight_ui(self, parent_container):
        frame = self._create_section_frame(parent_container, "自动飞行", 3)
        frame.grid_columnconfigure(2, weight=1) # Button expands

        # 四方飞行
        ctk.CTkLabel(frame, text="边长:", font=self.font_main).grid(row=0, column=0, padx=(self.padding,0), pady=self.padding, sticky="e")
        self.side_length_entry = ctk.CTkEntry(frame, width=80, font=self.font_main, corner_radius=self.corner_radius); self.side_length_entry.insert(0, "100")
        self.side_length_entry.grid(row=0, column=1, padx=5, pady=self.padding)

        # Radio buttons in their own sub-frame for better grouping
        # radio_frame = ctk.CTkFrame(frame, fg_color="transparent")
        # radio_frame.grid(row=0, column=2, padx=self.padding, pady=self.padding/2, sticky="ew")

        self.unit_var = tk.StringVar(value="distance")
        # ctk.CTkRadioButton(radio_frame, text="时间 (s)", variable=self.unit_var, value="time", font=self.font_main, corner_radius=self.corner_radius).pack(side="left", padx=(0,10))
        # ctk.CTkRadioButton(radio_frame, text="距离 (cm)", variable=self.unit_var, value="distance", font=self.font_main, corner_radius=self.corner_radius).pack(side="left")

        ctk.CTkButton(frame, text="飞行正方形路径", command=self.action_square_flight, height=self.button_height, font=self.font_main, corner_radius=self.corner_radius).grid(row=0, column=2, padx=(0, self.padding), pady=self.padding, sticky="ew")

        # 瞄准飞行
        ctk.CTkLabel(frame, text="瞄准时间：", font=self.font_main).grid(row=1, column=0, padx=(self.padding,0), pady=self.padding, sticky="e")
        self.aim_time_entry = ctk.CTkEntry(frame, width=80, font=self.font_main, corner_radius=self.corner_radius); self.aim_time_entry.insert(0, "10")
        self.aim_time_entry.grid(row=1, column=1, padx=5, pady=self.padding)

        ctk.CTkButton(frame, text="飞行正方形路径（瞄准）", command=self.action_square_aim_flight, height=self.button_height, font=self.font_main, corner_radius=self.corner_radius).grid(row=1, column=2, padx=(0, self.padding), pady=self.padding, sticky="ew")

        ctk.CTkButton(
            frame, text="安全闭环调试（20cm）", command=self.action_safe_loop_debug,
            height=self.button_height, font=self.font_main, corner_radius=self.corner_radius
        ).grid(row=2, column=0, columnspan=3, padx=self.padding, pady=self.padding, sticky="ew")

    # --- 飞行路径绘制相关方法 ---
    def update_flight_path(self, x, y, z, heading=None):
        """更新实时 3D 飞行轨迹场景。"""
        if (not hasattr(self, 'main_interface_created') or not self.main_interface_created or 
            self.cleanup_in_progress or not self.gui_active):
            return
            
        try:
            # Check if canvas still exists
            if not hasattr(self, 'canvas') or self.canvas is None:
                return
                
            # Add new path point
            self.flight_path_data['x'].append(x)
            self.flight_path_data['y'].append(y)
            self.flight_path_data['z'].append(z)
            self.flight_path_data['timestamps'].append(time.time())

            current_sample = (
                round(float(x), 1),
                round(float(y), 1),
                round(float(z), 1),
                round(float(heading), 1) if isinstance(heading, (int, float)) else heading,
            )
            now = time.time()
            if (
                self._last_path_sample == current_sample
                or now - self._last_path_ui_update < self._path_ui_interval_s
            ):
                self._last_path_sample = current_sample
                return
            self._last_path_sample = current_sample
            self._last_path_ui_update = now

            self._update_target_path_from_controller()

            xs = self.flight_path_data['x']
            ys = self.flight_path_data['y']
            zs = self.flight_path_data['z']
            scene_xs, scene_ys, scene_zs = self._local_many_to_scene(xs, ys, zs)
            field_x, field_y, field_z = self._local_to_field(x, y, z)
            takeoff_z = self._get_takeoff_point()[2]
            scene_x, scene_y, scene_z = self._map_to_scene(field_x, field_y, field_z)
            floor_x, floor_y, floor_z = self._map_to_scene(field_x, field_y, takeoff_z)
            self.path_line.set_data_3d(scene_xs, scene_ys, scene_zs)
            self.path_glow.set_data_3d(scene_xs, scene_ys, scene_zs)
            sample_step = max(1, len(scene_xs) // 70)
            sample_xs = scene_xs[::sample_step]
            sample_ys = scene_ys[::sample_step]
            sample_zs = scene_zs[::sample_step]
            if scene_xs and (not sample_xs or sample_xs[-1] != scene_xs[-1]):
                sample_xs.append(scene_xs[-1])
                sample_ys.append(scene_ys[-1])
                sample_zs.append(scene_zs[-1])
            self.path_points._offsets3d = (sample_xs, sample_ys, sample_zs)
            self.current_pos.set_data_3d([scene_x], [scene_y], [scene_z])
            self.ground_pos._offsets3d = ([floor_x], [floor_y], [floor_z])
            self.altitude_line.set_data_3d([floor_x, scene_x], [floor_y, scene_y], [floor_z, scene_z])

            self._draw_drone_model(scene_x, scene_y, scene_z, self._scene_heading(heading))
            if not self.path_manual_view:
                self._autoscale_3d_view()
            self.path_hud.set_text(
                f"局部 X {x:.1f} Y {y:.1f} Z {z:.1f} cm  |  场地 X {field_x:.1f} Y {field_y:.1f}  点数 {len(xs)}"
            )
            
            # Refresh canvas
            canvas = self.canvas
            if canvas is not None:
                canvas.draw_idle()
            
        except Exception as e:
            if not self.cleanup_in_progress:
                print(f"更新飞行路径时出错: {e}")
    
    def reset_path_view(self):
        """重置飞行路径视图"""
        if not hasattr(self, 'main_interface_created') or not self.main_interface_created:
            return

        self.path_manual_view = False
        self.path_drag_state = None
        self.ax.view_init(elev=35, azim=-90)
        self._autoscale_3d_view()
        
        canvas = self.canvas
        if canvas is not None:
            canvas.draw()
    
    def clear_flight_path(self):
        """清除飞行路径数据"""
        if not hasattr(self, 'main_interface_created') or not self.main_interface_created:
            return
            
        self.flight_path_data = {
            'x': [],
            'y': [],
            'z': [],
            'timestamps': []
        }
        self.target_path_data = {
            'x': [],
            'y': [],
            'z': []
        }
        self.path_manual_view = False
        self.path_drag_state = None
        self.last_target_location = None
        self.path_line.set_data_3d([], [], [])
        self.path_glow.set_data_3d([], [], [])
        self.ideal_line.set_data_3d([], [], [])
        self.path_points._offsets3d = ([], [], [])
        self.current_pos.set_data_3d([], [], [])
        self.ground_pos._offsets3d = ([], [], [])
        self.target_pos.set_data_3d([], [], [])
        self.altitude_line.set_data_3d([], [], [])
        takeoff_x, takeoff_y, takeoff_z = self._get_takeoff_point()
        drone_x, drone_y, drone_z = self._map_to_scene(takeoff_x, takeoff_y, takeoff_z)
        self._draw_drone_model(drone_x, drone_y, drone_z, self._scene_heading(0))
        self.path_hud.set_text(
            f"局部 X 0.0 Y 0.0 Z 0.0 cm  |  场地 X {takeoff_x:.1f} Y {takeoff_y:.1f}  点数 0"
        )

        # 恢复默认视图
        self.ax.view_init(elev=35, azim=-90)
        self._set_default_map_view()
        
        canvas = self.canvas
        if canvas is not None:
            canvas.draw()
    
    def save_flight_path_image(self):
        """保存当前飞行路径图像"""
        if not hasattr(self, 'main_interface_created') or not self.main_interface_created:
            return
            
        import datetime
        filename = f"flight_path_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        messagebox.showinfo("保存成功", f"飞行路径图像已保存为：\n{filename}")
    
    # --- 视频流处理相关方法 ---
    def start_video_stream(self):
        """启动视频流处理"""
        if not hasattr(self, 'main_interface_created') or not self.main_interface_created or self.cleanup_in_progress:
            return
                
        if not self.video_stream_active:
            self.video_stream_active = True
            self._set_video_stream_button_state(True)
            if hasattr(self, 'video_status_label') and self.video_status_label.winfo_exists():
                self.video_status_label.configure(text="视频流已开启")
            
            # 创建动画
            def update_frame(frame_num):
                if not self.drone.status["cam_stream"] or not self.gui_active or not self.video_stream_active or self.cleanup_in_progress:
                    return [self.video_img]
                    
                try:
                    if not self.image_raw_queue.empty():
                        frame = self.image_raw_queue.get_nowait()
                        if frame is not None:
                            # 调整大小并转换颜色空间
                            frame = cv2.resize(frame, (self.image_width, self.image_height))  # type: ignore[attr-defined]
                            # cv2.imwrite("temp_frame.jpg", frame)  # 保存临时帧用于调试
                            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            show_processed_frame = (
                                self.video_stream_show_target_frame
                                or self.video_stream_show_distance_frame
                            )
                            if not show_processed_frame: # 如果不显示处理后的帧
                                self.video_img.set_data(frame)
                            else:                                       # 显示视觉处理结果
                                # 显示目标检测或单目测距处理后的帧
                                if self.frame_queue.empty(): # 如果没有处理后的帧
                                    now = time.time()
                                    if now - self._last_video_status_update > 1.0:
                                        self._last_video_status_update = now
                                        self.main_status_label.configure(text="视觉处理：等待检测结果", text_color=self._get_status_color("orange"))
                                    self.video_img.set_data(frame)
                                else:
                                    target_frame = self.frame_queue.get_nowait()
                                    if target_frame is not None:
                                        target_frame = cv2.resize(target_frame, (self.image_width, self.image_height))  # type: ignore[attr-defined]
                                        # cv2.imwrite("temp_target.jpg", target_frame)  # 保存临时目标帧用于调试
                                        self.video_img.set_data(target_frame)
                                    else:
                                        # 如果没有目标帧，使用原始帧
                                        now = time.time()
                                        if now - self._last_video_status_update > 1.0:
                                            self._last_video_status_update = now
                                            self.main_status_label.configure(text="目标检测：未检测到目标", text_color=self._get_status_color("orange"))
                                        self.video_img.set_data(frame)

                    return [self.video_img]
                except Exception as e:
                    if not self.cleanup_in_progress:
                        print(f"视频帧更新错误: {e}")
                    return [self.video_img]
            
            try:
                self.video_animation = FuncAnimation(
                    self.video_fig, 
                    update_frame, 
                    interval=150,
                    blit=True,
                    cache_frame_data=False
                )
                video_canvas = self.video_canvas
                if video_canvas is not None:
                    video_canvas.draw()
            except Exception as e:
                print(f"创建视频动画时出错: {e}")
                self.video_stream_active = False
    
    def stop_video_stream(self):
        """停止视频流处理"""
        self.video_stream_active = False
        self._set_video_stream_button_state(False)
        
        if hasattr(self, 'video_animation') and self.video_animation is not None:
            try:
                event_source = getattr(self.video_animation, "event_source", None)
                if event_source is not None:
                    event_source.stop()
                self.video_animation = None
            except:
                pass
        
        try:
            if hasattr(self, 'video_status_label') and self.video_status_label.winfo_exists():
                self.video_status_label.configure(text="视频流已关闭")
            
            if hasattr(self, 'video_img') and hasattr(self, 'video_canvas'):
                black_frame = np.zeros((self.image_height, self.image_width, 3), dtype="uint8")
                self.video_img.set_data(black_frame)
                video_canvas = self.video_canvas
                if video_canvas is not None:
                    video_canvas.draw()
        except:
            pass

    # --- 动作方法 ---
    def _set_video_stream_button_state(self, active):
        if not hasattr(self, "video_stream_button"):
            return
        try:
            if self.video_stream_button.winfo_exists():
                self.video_stream_button.configure(text="关闭视频流" if active else "打开视频流")
        except Exception:
            pass

    def _run_drone_action_in_thread(self, action_func, *args, **kwargs):
        thread = threading.Thread(target=action_func, args=args, kwargs=kwargs, daemon=True)
        thread.start()

    def action_connect_drone(self):
        ip_text = self.ip_entry.get().strip()
        ip = ip_text if ip_text else None
        self.connect_button.configure(text="连接中...", state="disabled")
        self.status_label.configure(text="状态: 正在连接...", text_color=self._get_status_color("orange"))
        self._run_drone_action_in_thread(self.drone.connect, ip)

    def action_start_services(self):
        self.main_status_label.configure(text="状态: 正在准备服务...", text_color=self._get_status_color("orange"))
        self._run_drone_action_in_thread(self.drone.start_background_services)

    def action_takeoff(self):
        self.main_status_label.configure(text="状态: 正在起飞...", text_color=self._get_status_color("orange"))
        self._run_drone_action_in_thread(self.drone.takeoff)

    def action_land(self):
        self.main_status_label.configure(text="状态: 正在降落...", text_color=self._get_status_color("orange"))
        self._run_drone_action_in_thread(self.drone.land)

    def action_safe_loop_debug(self):
        confirmed = messagebox.askyesno(
            "安全确认",
            "将执行自动起飞与20cm小范围闭环调试：高度70cm，水平位移不超过20cm。\n"
            "请确认场地无人、无人机周围无遮挡，并准备随时点击降落。"
        )
        if not confirmed:
            return
        self.clear_flight_path()
        self.main_status_label.configure(text="状态: 安全闭环调试启动...", text_color=self._get_status_color("orange"))
        self._run_drone_action_in_thread(self._safe_loop_debug_sequence)

    def _safe_loop_debug_sequence(self):
        safe_plan = [
            [0, 0, 70, 0],
            [20, 0, 70, 0],
            [0, 0, 70, 0],
            [0, 20, 70, 0],
            [0, 0, 70, 0],
        ]
        for point in safe_plan:
            x, y, z = point[:3]
            if abs(x) > 20 or abs(y) > 20 or z < 50 or z > 100:
                print(f"安全闭环调试目标超限，已取消: {point}")
                return

        if not self.drone.status.get("connected"):
            print("安全闭环调试取消：无人机未连接")
            return

        self.drone.start_background_services()
        time.sleep(0.8)
        self.drone.takeoff()
        time.sleep(4.0)

        if self.drone.controller is None:
            print("安全闭环调试取消：控制器未初始化")
            return

        print("安全闭环调试开始：目标点限制在水平20cm、高度70cm")
        self.drone.execute_fly_plan(safe_plan)

    def action_move_to_target(self):
        try:
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            z = float(self.z_entry.get())
            self.main_status_label.configure(text=f"状态: 正在移动到 ({x},{y},{z})...", text_color=self._get_status_color("orange"))
            self._run_drone_action_in_thread(self.drone.move_to_global_target, x, y, z)
        except ValueError:
            messagebox.showerror("输入错误", "坐标必须为有效数字")

    def action_rotation(self):
        try:
            rotate_degrees = int(self.rotation_entry.get())
            self.main_status_label.configure(text=f"状态: 正在旋转 {rotate_degrees}°...", text_color=self._get_status_color("orange"))
            self._run_drone_action_in_thread(self.drone.set_rotation, rotate_degrees)
        except ValueError:
            messagebox.showerror("输入错误", "旋转角度必须为有效整数")

    def action_set_heading(self):
        try:
            heading = int(self.heading_entry.get())
            self.main_status_label.configure(text=f"状态: 正在设置航向到 {heading}°...", text_color=self._get_status_color("orange"))
            self._run_drone_action_in_thread(self.drone.set_heading, heading)
        except ValueError:
            messagebox.showerror("输入错误", "航向角度必须为有效整数")

    ## --- 手动控制Action ---    
    # 相机俯仰控制动作
    def action_camera_pitch_up(self):
        try:
            step = int(self.camera_step_entry.get())
            self.main_status_label.configure(text=f"状态: 相机向上调整 {step}°...", text_color=self._get_status_color("orange"))
            self._run_drone_action_in_thread(self.drone.set_camera_relative_pitch, step)  # 正值表示向上
        except ValueError:
            messagebox.showerror("输入错误", "相机步长必须为有效整数")
    
    def action_camera_pitch_down(self):
        try:
            step = int(self.camera_step_entry.get())
            self.main_status_label.configure(text=f"状态: 相机向下调整 {step}°...", text_color=self._get_status_color("orange"))
            self._run_drone_action_in_thread(self.drone.set_camera_relative_pitch, -step)  # 负值表示向下
        except ValueError:
            messagebox.showerror("输入错误", "相机步长必须为有效整数")
    
    # 航向控制动作
    def action_heading_left(self):
        try:
            step = int(self.heading_step_entry.get())
            self.main_status_label.configure(text=f"状态: 航向左转 {step}°...", text_color=self._get_status_color("orange"))
            self._run_drone_action_in_thread(self.drone.set_rotation, -step)  # 负值表示左转
        except ValueError:
            messagebox.showerror("输入错误", "航向步长必须为有效整数")
    
    def action_heading_right(self):
        try:
            step = int(self.heading_step_entry.get())
            self.main_status_label.configure(text=f"状态: 航向右转 {step}°...", text_color=self._get_status_color("orange"))
            self._run_drone_action_in_thread(self.drone.set_rotation, step)  # 正值表示右转
        except ValueError:
            messagebox.showerror("输入错误", "航向步长必须为有效整数")
    
    # 无人机移动控制动作 - 使用本地坐标系
    def action_move_forward(self):
        try:
            step = int(self.movement_step_entry.get())
            self.main_status_label.configure(text=f"状态: 向前移动 {step}cm...", text_color=self._get_status_color("orange"))
            # 在本地坐标系中，前进是y轴正方向
            self._run_drone_action_in_thread(self.drone.move_to_local_target, 0, step, 0)
        except ValueError:
            messagebox.showerror("输入错误", "移动步长必须为有效整数")
    
    def action_move_backward(self):
        try:
            step = int(self.movement_step_entry.get())
            self.main_status_label.configure(text=f"状态: 向后移动 {step}cm...", text_color=self._get_status_color("orange"))
            # 在本地坐标系中，后退是y轴负方向
            self._run_drone_action_in_thread(self.drone.move_to_local_target, 0, -step, 0)
        except ValueError:
            messagebox.showerror("输入错误", "移动步长必须为有效整数")
    
    def action_move_left(self):
        try:
            step = int(self.movement_step_entry.get())
            self.main_status_label.configure(text=f"状态: 向左移动 {step}cm...", text_color=self._get_status_color("orange"))
            # 在本地坐标系中，左移是x轴负方向
            self._run_drone_action_in_thread(self.drone.move_to_local_target, -step, 0, 0)
        except ValueError:
            messagebox.showerror("输入错误", "移动步长必须为有效整数")
    
    def action_move_right(self):
        try:
            step = int(self.movement_step_entry.get())
            self.main_status_label.configure(text=f"状态: 向右移动 {step}cm...", text_color=self._get_status_color("orange"))
            # 在本地坐标系中，右移是x轴正方向
            self._run_drone_action_in_thread(self.drone.move_to_local_target, step, 0, 0)
        except ValueError:
            messagebox.showerror("输入错误", "移动步长必须为有效整数")
    
    def action_move_up(self):
        try:
            step = int(self.movement_step_entry.get())
            self.main_status_label.configure(text=f"状态: 向上移动 {step}cm...", text_color=self._get_status_color("orange"))
            # 上升是z轴正方向
            self._run_drone_action_in_thread(self.drone.move_to_local_target, 0, 0, step)
        except ValueError:
            messagebox.showerror("输入错误", "移动步长必须为有效整数")
    
    def action_move_down(self):
        try:
            step = int(self.movement_step_entry.get())
            self.main_status_label.configure(text=f"状态: 向下移动 {step}cm...", text_color=self._get_status_color("orange"))
            # 下降是z轴负方向
            self._run_drone_action_in_thread(self.drone.move_to_local_target, 0, 0, -step)
        except ValueError:
            messagebox.showerror("输入错误", "移动步长必须为有效整数")

    ## --- 目标检测与瞄准 ---
    def action_detect(self):
        if not self.video_stream_show_target_frame:
            self.main_status_label.configure(text="状态: 正在进行目标检测...", text_color=self._get_status_color("orange"))
            self.video_stream_show_target_frame = True # 与 self.start_video_stream 方法有关
            self.drone.flag_cam_detect = True # 与 self.drone._capture_image_loop 方法有关
        else:
            self.main_status_label.configure(text="状态: 已停止目标检测", text_color=self._get_status_color("orange"))
            self.video_stream_show_target_frame = False
            self.drone.flag_cam_detect = False # 与 self.drone._capture_image_loop 方法有关

    def action_apply_monocular_reference(self):
        try:
            diameter = float(self.distance_diameter_entry.get())
            if diameter <= 0:
                raise ValueError()

            self.drone.set_monocular_reference_size(diameter)
            self.main_status_label.configure(
                text=f"状态: 红色圆片直径 {diameter:.1f}cm",
                text_color=self._get_status_color("orange"),
            )
            return True
        except ValueError:
            messagebox.showerror("输入错误", "红色圆形纸片直径必须为正数")
            return False

    def action_toggle_monocular_distance(self):
        enable = self.monocular_distance_var.get()
        if enable:
            if not self.video_stream_active:
                self.monocular_distance_var.set(False)
                messagebox.showwarning("需要视频流", "请先开启视频流，再启动单目测距")
                return
            if not self.action_apply_monocular_reference():
                self.monocular_distance_var.set(False)
                return

        self.video_stream_show_distance_frame = enable
        self.drone.set_monocular_distance_enabled(enable)
        status_text = "正在测距红色圆形纸片" if enable else "已停止单目视觉测距"
        self.main_status_label.configure(
            text=f"状态: {status_text}",
            text_color=self._get_status_color("orange"),
        )

    def action_toggle_red_circle_laser_tracking(self):
        enable = self.red_circle_track_var.get()
        if enable:
            if not self.video_stream_active:
                self.red_circle_track_var.set(False)
                messagebox.showwarning("需要视频流", "请先开启视频流，再启动红圆激光跟踪")
                return
            if not self.action_apply_monocular_reference():
                self.red_circle_track_var.set(False)
                return
            if not self.action_apply_red_circle_laser_offset():
                self.red_circle_track_var.set(False)
                return

            self.monocular_distance_var.set(True)
            self.video_stream_show_distance_frame = True
            self.drone.set_monocular_distance_enabled(True)
            self.laser_var.set(False)

        success = self.drone.set_red_circle_laser_tracking_enabled(enable)
        if not success:
            self.red_circle_track_var.set(False)
            return

        self.red_circle_laser_tracking = enable
        status_text = "正在红圆激光跟踪" if enable else "已停止红圆激光跟踪"
        self.main_status_label.configure(
            text=f"状态: {status_text}",
            text_color=self._get_status_color("orange"),
        )

    def action_apply_red_circle_laser_offset(self):
        try:
            offset_x = float(self.laser_offset_x_entry.get())
            offset_y = float(self.laser_offset_y_entry.get())
            self.drone.set_red_circle_laser_offset(offset_x, offset_y)
            self.main_status_label.configure(
                text=f"状态: 激光偏移 X {offset_x:+.0f}px, Y {offset_y:+.0f}px",
                text_color=self._get_status_color("orange"),
            )
            return True
        except ValueError:
            messagebox.showerror("输入错误", "激光偏移 X/Y 必须为数字")
            return False

    def action_toggle_aim_target(self):
        if not self.laser_aim_target:
            self.main_status_label.configure(text="状态: 正在瞄准目标...", text_color=self._get_status_color("orange"))
            # self.drone._aim_ready = True
            self.laser_aim_target = True
            self._run_drone_action_in_thread(self.drone.resume_aim_target)
        else:
            self.laser_aim_target = False
            self.main_status_label.configure(text="状态: 已停止瞄准目标", text_color=self._get_status_color("orange"))
            self._run_drone_action_in_thread(self.drone.pause_aim_target)
            # self.drone._aim_ready = False

    ## --- 自动飞行相关方法 ---
    def action_square_flight(self):
        try:
            side = float(self.side_length_entry.get())
            unit = self.unit_var.get()
            self.main_status_label.configure(text=f"状态: 正在开始正方形飞行 (边长: {side} {unit})...", text_color=self._get_status_color("orange"))
            self._run_drone_action_in_thread(self.drone.square_flight, side, unit, step_callback=self.drone.set_rotation)
        except ValueError:
            messagebox.showerror("输入错误", "边长必须为有效数字")

    def action_square_aim_flight(self):
        def completion_callback():
            self.drone.flag_cam_detect = True
            self.video_stream_show_target_frame = True
            self.laser_aim_target = True
            self._run_drone_action_in_thread(self.drone.pause_aim_target)

        try:
            self.drone.flag_cam_detect = True
            self.video_stream_show_target_frame = True
            self.laser_aim_target = False
            self._run_drone_action_in_thread(self.drone.pause_aim_target)

            side = float(self.side_length_entry.get())
            unit = self.unit_var.get()
            time = float(self.aim_time_entry.get())
            self.main_status_label.configure(text=f"状态: 正在开始正方形飞行【含激光瞄准】 (边长: {side} {unit})...", text_color=self._get_status_color("orange"))
            self._run_drone_action_in_thread(self.drone.square_aim_flight, side, unit, time, step_callback=self.drone.set_rotation, completion_callback=completion_callback)
        except ValueError:
            messagebox.showerror("输入错误", "边长和瞄准时间必须为有效数字")

    def action_toggle_laser(self):
        enable = self.laser_var.get()
        if enable and self.red_circle_laser_tracking:
            self.red_circle_track_var.set(False)
            self.action_toggle_red_circle_laser_tracking()
        action_text = "启用" if enable else "禁用"
        self.main_status_label.configure(text=f"状态: {action_text} 激光...", text_color=self._get_status_color("orange"))
        self._run_drone_action_in_thread(self.drone.toggle_laser, enable)

    def action_capture_image_stream(self):
        if not self.video_stream_active:
            self.main_status_label.configure(text="状态: 正在开启视频流...", text_color=self._get_status_color("orange"))
            self._run_drone_action_in_thread(self.drone.start_image_stream, self.image_raw_queue, self.frame_queue)
            self.start_video_stream()
        else:
            self.main_status_label.configure(text="状态: 正在关闭视频流...", text_color=self._get_status_color("orange"))
            self.stop_video_stream()
            self._run_drone_action_in_thread(self.drone.stop_image_stream)
        # else:
        #     self.main_status_label.configure(text="状态: 正在关闭视频流...", text_color=self._get_status_color("orange"))
        #     self._run_drone_action_in_thread(self.drone.stop_image_stream)
        #     self.stop_video_stream()

    # --- UI 更新和关闭 ---
    def _get_status_color(self, color_name: str):
        """返回主题适配的颜色"""
        is_dark_mode = ctk.get_appearance_mode() == "Dark"
        colors = {
            "green": ("#2ECC71", "#27AE60") if not is_dark_mode else ("#58D68D", "#2ECC71"),
            "orange": ("#F39C12", "#D35400") if not is_dark_mode else ("#F5B041", "#F39C12"),
            "red": ("#E74C3C", "#C0392B") if not is_dark_mode else ("#EC7063", "#E74C3C"),
            "grey": ("#95A5A6", "#7F8C8D") if not is_dark_mode else ("#BDC3C7", "#95A5A6"),
            "blue_text": ("#3498DB", "#5DADE2")
        }
        return colors.get(color_name.lower(), ("#000000", "#FFFFFF"))[1 if is_dark_mode else 0]

    def update_status_display_from_callback(self, drone_status: dict):
        if not self.gui_active or self.cleanup_in_progress: 
            return
        self._pending_status_update = drone_status.copy()
        if self._status_update_after_id is None:
            self._status_update_after_id = self._schedule_callback(
                self._status_update_interval_ms,
                self._flush_status_update,
            )


    def _flush_status_update(self):
        if self._status_update_after_id in self.scheduled_callbacks:
            self.scheduled_callbacks.remove(self._status_update_after_id)
        self._status_update_after_id = None
        drone_status = self._pending_status_update
        self._pending_status_update = None
        if drone_status is not None:
            self._do_update_ui(drone_status)

    def _do_update_ui(self, drone_status: dict):
        if not self.gui_active or self.cleanup_in_progress: 
            return

        try:
            msg = drone_status.get("message", "Unknown Status")
            self.connected = drone_status.get("connected", False)
            current_status_text = f"状态: {msg}"

            # Update connection interface status (if still visible)
            if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                if self.connected:
                    self.status_label.configure(text=current_status_text, text_color=self._get_status_color("green"))
                    self.connect_button.configure(text="已连接", state="disabled", fg_color=self._get_status_color("grey"))
                    
                    # Switch to main interface if first connection and not already created
                    if not hasattr(self, 'main_interface_shown') or not self.main_interface_shown:
                        self.main_interface_shown = True
                        self._schedule_callback(500, self.show_main_interface)
                else:
                    if "connecting" in msg.lower():
                        self.status_label.configure(text=current_status_text, text_color=self._get_status_color("orange"))
                    elif "failed" in msg.lower() or "error" in msg.lower() or "disconnected" in msg.lower():
                        self.status_label.configure(text=current_status_text, text_color=self._get_status_color("red"))
                        self.connect_button.configure(text="连接无人机", state="normal")
                    else:
                        self.status_label.configure(text=current_status_text, text_color=self._get_status_color("grey"))
                        self.connect_button.configure(text="连接无人机", state="normal")

            # Update other status info safely
            battery = drone_status.get("battery_level", "Unknown")
            battery_text = f"电池: {battery}%" if isinstance(battery, (int, float)) else f"电池: {battery}"
            
            loc = drone_status.get("location", ["N/A", "N/A"])
            height = drone_status.get("height", "N/A")
            pos_str = f"位置: X:{loc[0]} Y:{loc[1]} Z:{height}"
            
            heading = drone_status.get("heading", "N/A")
            heading_text = f"航向: {heading:.1f}°" if isinstance(heading, float) else f"航向: {heading}"

            # Update connection interface labels if they exist
            if hasattr(self, 'battery_label') and self.battery_label.winfo_exists():
                self.battery_label.configure(text=battery_text, text_color=self._get_status_color("blue_text"))
            if hasattr(self, 'position_label') and self.position_label.winfo_exists():
                self.position_label.configure(text=pos_str, text_color=self._get_status_color("blue_text"))
            if hasattr(self, 'heading_label') and self.heading_label.winfo_exists():
                self.heading_label.configure(text=heading_text, text_color=self._get_status_color("blue_text"))
            
            # Update main interface labels if they exist
            if hasattr(self, 'main_interface_created') and self.main_interface_created:
                if hasattr(self, 'main_status_label') and self.main_status_label.winfo_exists():
                    self.main_status_label.configure(text=current_status_text, text_color=self._get_status_color("green" if self.connected else "grey"))
                if hasattr(self, 'main_battery_label') and self.main_battery_label.winfo_exists():
                    self.main_battery_label.configure(text=battery_text, text_color=self._get_status_color("blue_text"))
                if hasattr(self, 'main_position_label') and self.main_position_label.winfo_exists():
                    self.main_position_label.configure(text=pos_str, text_color=self._get_status_color("blue_text"))
                if hasattr(self, 'main_heading_label') and self.main_heading_label.winfo_exists():
                    self.main_heading_label.configure(text=heading_text, text_color=self._get_status_color("blue_text"))
                
                # Update flight path if we have valid position data
                if loc[0] != "N/A" and loc[1] != "N/A" and height != "N/A":
                    try:
                        x, y, z = float(loc[0]), float(loc[1]), float(height)
                        self.update_flight_path(x, y, z, heading)
                    except (ValueError, TypeError):
                        pass

        except Exception as e:
            print(f"状态更新时发生错误: {e}")

    def _schedule_callback(self, delay, callback, *args):
        """Helper method to track scheduled callbacks for proper cleanup"""
        if self.cleanup_in_progress or not self.gui_active:
            return None
        try:
            callback_id = self.root.after(delay, callback, *args)
            self.scheduled_callbacks.append(callback_id)
            return callback_id
        except:
            return None

    def _cancel_all_callbacks(self):
        """Cancel all tracked callbacks safely"""
        for callback_id in self.scheduled_callbacks:
            try:
                if callback_id:
                    self.root.after_cancel(callback_id)
            except:
                pass
        self.scheduled_callbacks.clear()


    def on_closing_window(self):
        if messagebox.askokcancel("退出确认", "您确定要退出 Hula Drone 控制界面吗？\n无人机将尝试安全着陆并保存数据。"):
            print("开始关闭应用程序...")
            
            # Set cleanup flag immediately
            self.cleanup_in_progress = True
            self.gui_active = False
            self.video_stream_active = False
            
            # Stop video stream first
            self.stop_video_stream()
            
            # Cancel all scheduled callbacks
            self._cancel_all_callbacks()
            
            # Update status safely
            try:
                if hasattr(self, 'main_interface_created') and self.main_interface_created:
                    if hasattr(self, 'main_status_label') and self.main_status_label.winfo_exists():
                        self.main_status_label.configure(text="状态: 正在退出，请稍候...", text_color=self._get_status_color("orange"))
                else:
                    if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                        self.status_label.configure(text="状态: 正在退出，请稍候...", text_color=self._get_status_color("orange"))
            except:
                pass

            # Force update the display
            try:
                self.root.update_idletasks()
            except:
                pass
            
            # Unregister drone callback safely
            try:
                if hasattr(self.drone, 'unregister_status_callback'):
                    self.drone.unregister_status_callback(self.update_status_display_from_callback)
            except:
                pass

            # Start drone shutdown in background
            exit_thread = threading.Thread(target=self.drone.graceful_exit, daemon=True)
            exit_thread.start()
            
            # Schedule window destruction with a delay
            self.root.after(1000, self._destroy_root_safely)

    def _destroy_root_safely(self):
        """Safely destroy the root window and clean up resources"""
        if not self.cleanup_in_progress:
            self.cleanup_in_progress = True
        
        print("正在安全关闭图形界面...")
        
        try:
            # Cancel any remaining callbacks
            self._cancel_all_callbacks()
            
            # Stop matplotlib animations
            if hasattr(self, 'video_animation') and self.video_animation is not None:
                event_source = getattr(self.video_animation, "event_source", None)
                if event_source is not None:
                    event_source.stop()
                self.video_animation = None
                
            # Close matplotlib figures
            if hasattr(self, 'video_fig'):
                plt.close(self.video_fig)
            if hasattr(self, 'fig'):
                if hasattr(self, 'canvas') and self.canvas is not None:
                    for cid in getattr(self, "path_interaction_cids", []):
                        try:
                            self.canvas.mpl_disconnect(cid)
                        except Exception:
                            pass
                    self.path_interaction_cids = []
                plt.close(self.fig)
                
            # Clear references to prevent callback issues
            if hasattr(self, 'video_canvas'):
                self.video_canvas = None
            if hasattr(self, 'canvas'):
                self.canvas = None
                
            # Destroy the root window
            if self.root and self.root.winfo_exists():
                self.root.quit()       # Exit the mainloop
                self.root.destroy()    # Destroy the window
                
        except Exception as e: 
            print(f"关闭图形界面时发生错误: {e}")
            # Force exit if normal cleanup fails
            import sys
            sys.exit(0)
        finally:
            print("图形界面已安全关闭。")
    
    def run_gui(self):
        self.gui_active = True
        self.root.mainloop()

# Main program entry point
if __name__ == "__main__":
    gui_root = ctk.CTk()
    app = HulaDroneGUI_CTk_Enhanced(gui_root)
    try:
        app.run_gui()
    except KeyboardInterrupt:
        print("\n检测到Ctrl+C，正在关闭应用程序...")
        if hasattr(app, 'gui_active') and app.gui_active: 
            app.on_closing_window()
    except Exception as e:
        print(f"在GUI主循环中发生未处理的异常: {e}")
    finally:
        print("应用程序主线程已完成。")
