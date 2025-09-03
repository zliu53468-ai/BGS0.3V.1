# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import json
import requests
import logging
import sys
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import io

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('BaccaratMonitor')

class DGMonitor:
    def __init__(self, config_file='config.json'):
        self.config = self.load_config(config_file)
        self.is_monitoring = False
        self.roadmap = []
        self.backend_url = os.environ.get('BACKEND_URL', 'http://localhost:5000')  # 从环境变量获取后端URL
        self.driver = None
        self.target_url = "https://new-dd-cloudfront.ywjxi.com/ddnewwap/index.html?token=cc910b189b244cc3b4dfa12aa03bbc83&language=tw&backUrl=https://art888.3a5168.com&type=5&return=dggw.vip"
        
    def load_config(self, config_file):
        """加载配置文件"""
        default_config = {
            "roadmap_region": {"x": 100, "y": 100, "width": 300, "height": 200},
            "check_interval": 5,
            "banker_color": [50, 50, 200],    # BGR格式的红色
            "player_color": [200, 50, 50],    # BGR格式的蓝色
            "tie_color": [50, 200, 50],       # BGR格式的绿色
            "color_tolerance": 30,
            "chrome_driver_path": "chromedriver",
            "headless": False
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            
        return default_config
    
    def save_config(self, config_file='config.json'):
        """保存配置文件"""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            return False
    
    def setup_driver(self):
        """设置Chrome浏览器驱动"""
        try:
            chrome_options = Options()
            if self.config.get("headless", False):
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # 设置中文语言
            chrome_options.add_argument("--lang=zh-TW")
            
            # 忽略证书错误
            chrome_options.add_argument("--ignore-certificate-errors")
            
            # 使用WebDriverManager自动管理驱动
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                driver_path = ChromeDriverManager().install()
            except:
                driver_path = self.config.get("chrome_driver_path", "chromedriver")
                
            self.driver = webdriver.Chrome(executable_path=driver_path, options=chrome_options)
            
            logger.info("Chrome浏览器驱动设置成功")
            return True
            
        except Exception as e:
            logger.error(f"设置浏览器驱动失败: {e}")
            return False
    
    def login(self):
        """导航到目标网址并等待页面加载"""
        try:
            logger.info(f"正在导航到目标网址: {self.target_url}")
            self.driver.get(self.target_url)
            
            # 等待页面加载完成
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            logger.info("页面加载成功")
            
            # 等待游戏界面加载
            time.sleep(10)
            
            # 尝试寻找游戏区域
            try:
                game_element = WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".game-container, .baccarat-table, .roadmap"))
                )
                logger.info("检测到游戏界面")
            except:
                logger.warning("未检测到明显的游戏元素，将继续尝试")
                
            return True
            
        except Exception as e:
            logger.error(f"导航到目标网址失败: {e}")
            return False
    
    def capture_screen(self):
        """捕获屏幕指定区域"""
        try:
            # 获取整个页面的截图
            screenshot = self.driver.get_screenshot_as_png()
            image = Image.open(io.BytesIO(screenshot))
            
            # 转换为OpenCV格式
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 裁剪指定区域
            region = self.config["roadmap_region"]
            x, y, width, height = region["x"], region["y"], region["width"], region["height"]
            
            # 确保区域在图像范围内
            height, width_total = opencv_image.shape[:2]
            if x + width > width_total:
                width = width_total - x
            if y + height > height:
                height = height - y
                
            if width <= 0 or height <= 0:
                logger.error("截图区域超出屏幕范围")
                return None
                
            cropped_image = opencv_image[y:y+height, x:x+width]
            return cropped_image
            
        except Exception as e:
            logger.error(f"截图失败: {e}")
            return None
    
    def analyze_image(self, image):
        """分析图像，识别牌路"""
        if image is None:
            return []
            
        results = []
        
        # 简化的识别逻辑：检查特定位置的特定颜色
        # 实际应用中，这里会是更复杂的CV算法
        
        height, width = image.shape[:2]
        
        # 检查5个预设点
        check_points = [
            (width//4, height//4),    # 左上区域
            (width//2, height//4),    # 中上区域
            (3*width//4, height//4),  # 右上区域
            (width//4, 3*height//4),  # 左下区域
            (3*width//4, 3*height//4) # 右下区域
        ]
        
        for point in check_points:
            x, y = point
            pixel_color = image[y, x]
            
            # 检查颜色匹配
            if self.is_color_match(pixel_color, self.config["banker_color"]):
                results.append("B")
            elif self.is_color_match(pixel_color, self.config["player_color"]):
                results.append("P")
            elif self.is_color_match(pixel_color, self.config["tie_color"]):
                results.append("T")
        
        return results
    
    def is_color_match(self, color1, color2):
        """检查两个颜色是否匹配（在容差范围内）"""
        return all(abs(c1 - c2) < self.config["color_tolerance"] 
                  for c1, c2 in zip(color1, color2))
    
    def send_to_backend(self, results, table_id):
        """将识别结果发送到后端"""
        try:
            response = requests.post(
                f"{self.backend_url}/api/update-roadmap",
                json={
                    "results": results,
                    "table_id": table_id
                },
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"发送数据到后端失败: {e}")
            return False
    
    def calibrate(self):
        """校准屏幕区域"""
        logger.info("请确保浏览器窗口已打开并显示游戏界面，5秒后开始校准...")
        time.sleep(5)
        
        # 获取整个页面的截图
        screenshot = self.driver.get_screenshot_as_png()
        image = Image.open(io.BytesIO(screenshot))
        image.save("calibration_screenshot.png")
        
        logger.info("已保存截图到 calibration_screenshot.png")
        logger.info("请手动测量牌路区域的坐标和尺寸，并更新config.json文件中的roadmap_region设置")
        
        return True
    
    def start(self, table_id="default"):
        """开始监控"""
        if not self.setup_driver():
            return False
            
        if not self.login():
            self.driver.quit()
            return False
            
        self.is_monitoring = True
        logger.info(f"开始监控牌桌 {table_id}")
        
        last_results = []
        
        while self.is_monitoring:
            try:
                # 捕获屏幕
                screenshot = self.capture_screen()
                
                if screenshot is None:
                    logger.error("无法获取截图，等待10秒后重试")
                    time.sleep(10)
                    continue
                
                # 分析图像
                current_results = self.analyze_image(screenshot)
                
                # 检测新结果
                new_results = []
                for result in current_results:
                    if result not in last_results[-3:]:  # 简单的去重逻辑
                        new_results.append(result)
                
                if new_results:
                    logger.info(f"检测到新结果: {new_results}")
                    # 发送到后端
                    if self.send_to_backend(new_results, table_id):
                        last_results.extend(new_results)
                
                time.sleep(self.config["check_interval"])
                
            except Exception as e:
                logger.error(f"监控过程中发生错误: {e}")
                time.sleep(10)  # 出错后等待更长时间
        
        # 停止监控后关闭浏览器
        self.driver.quit()
        return True
    
    def stop(self):
        """停止监控"""
        self.is_monitoring = False
        logger.info("监控已停止")

if __name__ == '__main__':
    monitor = DGMonitor()
    
    # 检查是否需要校准
    if not os.path.exists('config.json'):
        logger.info("首次运行，需要先进行校准")
        if monitor.setup_driver() and monitor.login():
            monitor.calibrate()
        else:
            logger.error("无法启动浏览器进行校准")
        exit(0)
    
    # 开始监控
    try:
        monitor.start("B01")  # 默认监控B01桌
    except KeyboardInterrupt:
        monitor.stop()
        logger.info("监控已由用户中断")