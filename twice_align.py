import grpc
from concurrent import futures
import PTZService_pb2
import PTZService_pb2_grpc
import DeviceIdentifyGPRCService_pb2
import DeviceIdentifyGPRCService_pb2_grpc
import json
import cv2
import numpy as np
import os
import time
from datetime import datetime
import time
import logging
import logging.handlers

class PTZClient:
    def __init__(self, host='localhost', port=10011):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = PTZService_pb2_grpc.PTZInterfaceStub(self.channel)

    def init(self):
        request = PTZService_pb2.void_request()
        response = self.stub.init(request)
        return response

    def get_version(self):
        request = PTZService_pb2.void_request()
        response = self.stub.getVersion(request)
        return response

    def get_bearing_val(self):
        request = PTZService_pb2.void_request()
        response = self.stub.getBearingVal(request)
        return response.bearingval

    def get_pitching_val(self):
        request = PTZService_pb2.void_request()
        response = self.stub.getPitchingVal(request)
        return response.pitchingval

    def is_bearing_inplace(self):
        request = PTZService_pb2.void_request()
        response = self.stub.isBearingInplace(request)
        return response

    def is_pitching_inplace(self):
        request = PTZService_pb2.void_request()
        response = self.stub.isPitchingInplace(request)
        return response

    def stop(self):
        request = PTZService_pb2.void_request()
        response = self.stub.stop(request)
        return response

    def down(self):
        request = PTZService_pb2.void_request()
        response = self.stub.down(request)
        return response

    def up(self):
        request = PTZService_pb2.void_request()
        response = self.stub.up(request)
        return response

    def left(self):
        request = PTZService_pb2.void_request()
        response = self.stub.left(request)
        return response

    def right(self):
        request = PTZService_pb2.void_request()
        response = self.stub.right(request)
        return response

    def upleft(self):
        request = PTZService_pb2.void_request()
        response = self.stub.upleft(request)
        return response

    def upright(self):
        request = PTZService_pb2.void_request()
        response = self.stub.upright(request)
        return response

    def downleft(self):
        request = PTZService_pb2.void_request()
        response = self.stub.downleft(request)
        return response

    def downright(self):
        request = PTZService_pb2.void_request()
        response = self.stub.downright(request)
        return response

    def set_positioning_speed(self, hv, vv):
        request = PTZService_pb2.PTZSpeed(hv=hv, vv=vv)
        response = self.stub.setPositioningSpeed(request)
        return response

    def set_cruising_speed(self, hv, vv):
        request = PTZService_pb2.PTZSpeed(hv=hv, vv=vv)
        response = self.stub.setCruisingSpeed(request)
        return response

    def set_bearing(self, bearingval):
        request = PTZService_pb2.PTZBearing(bearingval=bearingval)
        response = self.stub.setBearing(request)
        return response

    def set_pitching(self, pitchingval):
        request = PTZService_pb2.PTZPitching(pitchingval=pitchingval)
        response = self.stub.setPitching(request)
        return response

    def set_bearing_and_pitching(self, bearingval, pitchingval):
        request = PTZService_pb2.PTZPose(bearingval=bearingval, pitchingval=pitchingval)
        response = self.stub.setBearingandPitching(request)
        return response

    def set_wiper_on(self):
        request = PTZService_pb2.void_request()
        response = self.stub.setWiperOn(request)
        return response

    def set_wiper_off(self):
        request = PTZService_pb2.void_request()
        response = self.stub.setWiperOff(request)
        return response

    def set_headlamp_on(self):
        request = PTZService_pb2.void_request()
        response = self.stub.setHeadlampOn(request)
        return response

    def set_headlamp_off(self):
        request = PTZService_pb2.void_request()
        response = self.stub.setHeadlampOff(request)
        return response

    def set_init_position(self):
        request = PTZService_pb2.void_request()
        response = self.stub.setInitPosition(request)
        return response

    def reboot(self):
        request = PTZService_pb2.void_request()
        response = self.stub.reboot(request)
        return response

    def set_zoom(self, zoomval):
        request = PTZService_pb2.VLZoom(zoomval=zoomval)
        response = self.stub.setZoom(request)
        return response

    def set_focus(self, focusval):
        request = PTZService_pb2.VLFocus(focusval=focusval)
        response = self.stub.setFocus(request)
        return response

    def is_zoom_inplace(self):
        request = PTZService_pb2.void_request()
        response = self.stub.isZoomInplace(request)
        return response

    def is_focus_inplace(self):
        request = PTZService_pb2.void_request()
        response = self.stub.isFocusInplace(request)
        return response

    def get_zoom(self):
        request = PTZService_pb2.void_request()
        response = self.stub.getZoom(request)
        return response.zoomval

    def get_focus(self):
        request = PTZService_pb2.void_request()
        response = self.stub.getFocus(request)
        return response.focusval

    def zoom_in(self):
        request = PTZService_pb2.void_request()
        response = self.stub.zoomIn(request)
        return response

    def zoom_out(self):
        request = PTZService_pb2.void_request()
        response = self.stub.zoomOut(request)
        return response

    def zoom_in_step(self):
        request = PTZService_pb2.void_request()
        response = self.stub.zoomInStep(request)
        return response

    def zoom_out_step(self):
        request = PTZService_pb2.void_request()
        response = self.stub.zoomOutStep(request)
        return response

    def focus_in(self):
        request = PTZService_pb2.void_request()
        response = self.stub.focusIn(request)
        return response

    def focus_out(self):
        request = PTZService_pb2.void_request()
        response = self.stub.focusOut(request)
        return response

    def set_focus_manual_mode(self):
        request = PTZService_pb2.void_request()
        response = self.stub.setFocusManualMode(request)
        return response

    def set_focus_auto_mode(self):
        request = PTZService_pb2.void_request()
        response = self.stub.setFocusAutoMode(request)
        return response

class DatedTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """自定义日志处理器，文件名包含当前日期"""
    def __init__(self, filename, when='midnight', interval=1, backupCount=0, 
                 encoding=None, delay=False, utc=False, atTime=None):
        # 生成带日期的文件名
        base, ext = os.path.splitext(filename)

        dated_filename = f"{base}_{datetime.now().strftime('%Y%m%d')}{ext}"
        
        # 调用父类初始化
        super().__init__(
            filename=dated_filename,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            utc=utc,
            atTime=atTime
        )
        
        # 存储基础文件名用于后续滚动
        self.base_filename = filename
        self.base, self.ext = base, ext
        self.current_date = datetime.now().date()
        
    def shouldRollover(self, record):
        """检查是否需要滚动日志（日期变化时）"""
        current_date = datetime.now().date()
        if current_date != self.current_date:
            self.current_date = current_date
            return True
        return False
        
    def doRollover(self):
        """执行日志滚动（创建新日期文件）"""
        # 关闭当前流
        if self.stream:
            self.stream.close()
            self.stream = None
            
        # 生成新日期文件名
        new_filename = f"{self.base}_{self.current_date.strftime('%Y%m%d')}{self.ext}"
        self.baseFilename = new_filename
        
        # 打开新文件
        if not self.delay:
            self.stream = self._open()
        
        # 处理备份文件
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)

# 检查并创建日志目录
log_directory = "log"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# 创建日志记录器
logger = logging.getLogger("my_app")
logger.setLevel(logging.DEBUG)

# 使用自定义的日期文件处理器
log_path = os.path.join(log_directory, "Flir.log")  # 日志文件路径
file_handler = DatedTimedRotatingFileHandler(
    filename=log_path,  # 日志文件保存路径
    when='midnight',     # 每天午夜检查
    backupCount=7,       # 保留7天日志
    encoding='utf-8'     # 解决中文乱码
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s'
))

        
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"目录 {directory_path} 不存在，已创建。")
    else:
        logger.info(f"目录 {directory_path} 已存在。")

dir_temp_img = "temp_img"
ensure_directory_exists(dir_temp_img)


def unevenLightCompensate(img, blockSize):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	average = np.mean(gray)

	rows_new = int(np.ceil(gray.shape[0] / blockSize))
	cols_new = int(np.ceil(gray.shape[1] / blockSize))

	blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
	for r in range(rows_new):
		for c in range(cols_new):
			rowmin = r * blockSize
			rowmax = (r + 1) * blockSize
			if (rowmax > gray.shape[0]):
				rowmax = gray.shape[0]
			colmin = c * blockSize
			colmax = (c + 1) * blockSize
			if (colmax > gray.shape[1]):
				colmax = gray.shape[1]

			imageROI = gray[rowmin:rowmax, colmin:colmax]
			temaver = np.mean(imageROI)
			blockImage[r, c] = temaver
	blockImage = blockImage - average
	blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
	gray2 = gray.astype(np.float32)
	dst = gray2 - blockImage2
	dst = dst.astype(np.uint8)
	dst = cv2.GaussianBlur(dst, (3, 3), 0)
	dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

	return dst

def Identify():
    file_path = "D:/FirstAlignment/2024-09-19/0.json"
    areas = []
    # template_path = 'D:/FirstAlignment/' + data['image_ID'] + '.jpg'
    template_path = "D:/FirstAlignment/2024-09-19/0.jpg"
    input_path = "D:/FirstAlignment/2024-09-19/4.jpg"
    magnification = 6555
    initHorizontalAngel = 144.35
    initVerticalAngel =2.98

    # test start 
    magnification = 4243
    initHorizontalAngel = 18.44
    initVerticalAngel =5.64
    # test over


    # 打开并读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    # 创建一个列表来存储所有的区域信息，每个区域信息是一个字典
    # 获取区域信息
    areas_info = data['areas']['areas_info']
    for area in areas_info:
        area_roi = area['area_ROI']
        meters = []
        # 获取每个区域中的仪表信息
        meters_info = area['groups']['groups_info'][0]['meters']['meters_info']
        # 将区域和其包含的仪表信息一起存储
        areas.append({
            'area': area_roi,
            'meters': meters
        })
        for meter in meters_info:
            meter_roi = meter['meter_ROI']
            meters.append(meter_roi)
            

    # 打印结果
    for area in areas:
        for meter in area['meters']:
            roi_x = meter['x']
            roi_y = meter['y']
            roi_w = meter['w']
            roi_h = meter['h']
            x1, y1, x2, y2 = int(roi_x), int(roi_y), int(roi_x)+int(roi_w), int(roi_y)+int(roi_h)
            img_template = cv2.imread(template_path)
            roi_template = img_template[y1:y2, x1:x2]
            
            
    x = np.array([0, 5785, 8381, 9944, 11066, 11897])
    y = np.array([0.03,0.0146,0.01,0.0081,0.0061,0.0037])

    # 使用numpy.polyfit拟合七项式
    coefficients = np.polyfit(x, y, 7)

    # 打印多项式系数
    # print("Coefficients:", coefficients)

    # 生成拟合曲线
    x_fit = np.linspace(min(x), max(x), 100)
    p = np.poly1d(coefficients)
    magnification_k = p(magnification)

    img = cv2.imread(input_path)
    roi_template = unevenLightCompensate(roi_template,20)
    img = unevenLightCompensate(img,20)
    res = cv2.matchTemplate(img, roi_template, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 绘制矩形框 
    top_left = max_loc
    bottom_right = (top_left[0] + roi_w, top_left[1] + roi_h)
    # 模版匹配后的仪表图像
    roi_img = img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]  
    cv2.imwrite('saved_image_path.jpg', roi_img)

    # print(top_left)
    horizontalPixelDifference =  top_left[0] - roi_x
    verticalPixelDifference = top_left[1] - roi_y

    # test start
    horizontalPixelDifference =  897
    verticalPixelDifference = 310
    # test over 

    # 计算水平
    horizontalAngleChange = magnification_k * horizontalPixelDifference
    horizontalAngleFinall = initHorizontalAngel - horizontalAngleChange
        
    if horizontalAngleFinall < 0:
        horizontalAngleFinall = 360 + horizontalAngleFinall
    if horizontalAngleFinall > 360:
        horizontalAngleFinall = horizontalAngleFinall - 360
    # 计算俯仰
    verticalAngleChange = magnification_k * verticalPixelDifference
    verticalAngleFinall = initVerticalAngel - verticalAngleChange
        
    if verticalAngleFinall < 0:
        verticalAngleFinall = 360 + verticalAngleFinall
    if verticalAngleFinall > 360:
        verticalAngleFinall = verticalAngleFinall - 360

    print(horizontalAngleFinall,verticalAngleFinall)

class DeviceIdentifyGPRCService(DeviceIdentifyGPRCService_pb2_grpc.DeviceIdentifyGPRCServiceServicer):
    def Identify(self, request, context):
        first_x_value = -1
        # 查询文件是否存在
        if os.path.exists('first_x_value.json'):
            with open('first_x_value.json', 'r') as file:
                data = json.load(file)
                first_x_value = data['first_x_value']
                first_y_value = data['first_y_value']
        else:
            pass
            # 创建文件
            # with open('first_x_value.json', 'w') as file:
            #     pass  # 创建空文件
        # 示例实现，返回模拟数据
        # 模板图像与输入图像路径
        if first_x_value < 0:
            #第一次对准
            logger.info('一次对准开启...')
            template_path = request.modelpath  #"D:/FirstAlignment/2024-09-19/0.jpg"
            input_path = request.filepath  #"D:/FirstAlignment/2024-09-19/4.jpg"
            # 获取json路径
            directory, filename_with_extension = os.path.split(template_path)
            filename, extension = os.path.splitext(filename_with_extension)
            new_extension = ".json"
            new_path = os.path.join(directory, filename + new_extension)
            file_path = new_path  #"D:/FirstAlignment/2024-09-19/0.json"
            logger.info(f'一次对准路径:')
            logger.info(input_path)
            logger.info(f'模板路径:')
            logger.info(template_path)
            logger.info(f'json路径:')
            logger.info(new_path)
            logger.info('开始获取云台参数...')
            areas = []
            # 获取俯仰角与放大倍率
            ''' 
            try:
                client = PTZClient()
                bearing_val = client.get_bearing_val()
                pitching_val = client.get_pitching_val()
                zoom_val = client.get_zoom()
                magnification = zoom_val
            except:
                logger.info('获取云台失败，请检查云台连接是否正确...')
    
            if float(bearing_val)+float(pitching_val)+float(magnification) == 0:
                time.sleep(0.2)
                try:
                    client = PTZClient()
                    bearing_val = client.get_bearing_val()
                    pitching_val = client.get_pitching_val()
                    zoom_val = client.get_zoom()
                    magnification = zoom_val
                except:
                    logger.info('获取云台失败，请检查云台连接是否正确...')
            if float(bearing_val)+float(pitching_val)+float(magnification) == 0:
                time.sleep(0.2)
                try:
                    client = PTZClient()
                    bearing_val = client.get_bearing_val()
                    pitching_val = client.get_pitching_val()
                    zoom_val = client.get_zoom()
                    magnification = zoom_val
                except:
                    logger.info('获取云台失败，请检查云台连接是否正确...')
            if float(bearing_val)+float(pitching_val)+float(magnification) == 0:
                time.sleep(0.2)
                try:
                    client = PTZClient()
                    bearing_val = client.get_bearing_val()
                    pitching_val = client.get_pitching_val()
                    zoom_val = client.get_zoom()
                    magnification = zoom_val
                except:
                    logger.info('获取云台失败，请检查云台连接是否正确...')
            if float(bearing_val)+float(pitching_val)+float(magnification) == 0:
                time.sleep(0.2)
                try:
                    client = PTZClient()
                    bearing_val = client.get_bearing_val()
                    pitching_val = client.get_pitching_val()
                    zoom_val = client.get_zoom()
                    magnification = zoom_val
                except:
                    logger.info('获取云台失败，请检查云台连接是否正确...')
            '''
            while True:
                try:
                    client = PTZClient()
                    bearing_val = client.get_bearing_val()
                    pitching_val = client.get_pitching_val()
                    # zoom_val = client.get_zoom()
                    # magnification = zoom_val
                except:
                    logger.exception('获取云台失败，请检查云台连接是否正确...')
                if float(bearing_val) == -1 or float(pitching_val) == -1:
                    logger.warning('云台获取数据异常，重新获取...')
                    time.sleep(0.2)
                else:
                    break
            
            initHorizontalAngel = float(bearing_val / 100)
            initVerticalAngel = float(pitching_val / 100)
            logger.info('云台参数获取成功！')
            # logger.info('放大倍率:')
            # logger.info(magnification)
            logger.info('水平角:')
            logger.info(bearing_val)
            logger.info('俯仰角:')
            logger.info(pitching_val)
            logger.info('开始解析json文档...')
            # # test start 
            # magnification = 4243
            # initHorizontalAngel = 18.44
            # initVerticalAngel =5.64
            # # test over


            # 打开并读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # 创建一个列表来存储所有的区域信息，每个区域信息是一个字典
            # 获取区域信息
            areas_info = data['areas']['areas_info']
            logger.info(len(areas_info))
            for area in areas_info:
                area_roi = area['area_ROI']
                meters = []
                # 获取每个区域中的仪表信息
                meters_info = area['groups']['groups_info'][0]['meters']['meters_info']
                # 将区域和其包含的仪表信息一起存储
                areas.append({
                    'area': area_roi,
                    'meters': meters
                })
                for meter in meters_info:
                    meter_roi = meter['meter_ROI']
                    meters.append(meter_roi)
                    

            # 打印结果
            for area in areas:
                for meter in area['meters']:
                    roi_x = meter['x']
                    roi_y = meter['y']
                    roi_w = meter['w']
                    roi_h = meter['h']
                    x1, y1, x2, y2 = int(roi_x), int(roi_y), int(roi_x)+int(roi_w), int(roi_y)+int(roi_h)
                    img_template = cv2.imread(template_path)
                    # roi_template = img_template[y1:y2, x1:x2]
                    
                    # logger.info(roi_x,roi_y,roi_w,roi_h)
                    
            logger.info('开始进行第一次对准...')
            try:
                img = cv2.imread(input_path)
            except Exception as e:
                logger.exception(f"发生错误: {e}")
                data = {"first_x_value": 0,"first_y_value": 0}
                with open('first_x_value.json', 'w') as file:
                    json.dump(data, file)
                return DeviceIdentifyGPRCService_pb2.identifyValue(mValue=0, valueType=0)
            # img = cv2.imread(input_path)
            cv2.imwrite(f'./{dir_temp_img}/{filename}_first_image.jpg', img)
            img_template = unevenLightCompensate(img_template,20)
            roi_template = img_template[y1:y2, x1:x2]
            img = unevenLightCompensate(img,20)
            res = cv2.matchTemplate(img, roi_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # 绘制矩形框
            top_left = max_loc
            bottom_right = (top_left[0] + roi_w, top_left[1] + roi_h)
            # 模版匹配后的仪表图像
            roi_img = img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]  
            cv2.imwrite(f'./{dir_temp_img}/{filename}_saved_image_path.jpg', roi_img)

            # logger.info(top_left)
            horizontalPixelDifference =  -round(top_left[0] - (960 - (roi_w / 2)))
            verticalPixelDifference = -round(top_left[1] - (540 - (roi_h / 2)))
            # logger.info('水平和垂直变化像素为:')
            # logger.info(f'{horizontalPixelDifference},{verticalPixelDifference}')
            # # test start
            # horizontalPixelDifference =  897
            # verticalPixelDifference = 310
            # # test over 

            # 计算水平
            # horizontalAngleChange = magnification_k * horizontalPixelDifference
            if horizontalPixelDifference > 0:
                horizontalAngleChange = 1
            else:
                horizontalAngleChange = -1
            horizontalAngleFinall = initHorizontalAngel + horizontalAngleChange
                
            if horizontalAngleFinall < 0:
                horizontalAngleFinall = 360 + horizontalAngleFinall
            if horizontalAngleFinall > 360:
                horizontalAngleFinall = horizontalAngleFinall - 360
            # # 计算俯仰
            
            if verticalPixelDifference > 0:
                verticalAngleChange = 1
            else:
                verticalAngleChange = -1
            verticalAngleFinall = initVerticalAngel + verticalAngleChange  
            if verticalAngleFinall < 0:
                verticalAngleFinall = 360 + verticalAngleFinall
            if verticalAngleFinall > 360:
                verticalAngleFinall = verticalAngleFinall - 360
            horizontalAngleFinall = int(round(horizontalAngleFinall * 100, 0))
            verticalAngleFinall = int(round(verticalAngleFinall * 100, 0))
            logger.info('第一次对准完成，云台调整到：')
            logger.info(f'{horizontalAngleFinall},{verticalAngleFinall}')
            data = {"first_x_value": top_left[0],"first_y_value": top_left[1]}
            with open('first_x_value.json', 'w') as file:
                json.dump(data, file)

        else:
            logger.info('第二次对准开启...')
            
            template_path = request.modelpath  #"D:/FirstAlignment/2024-09-19/0.jpg"
            input_path = request.filepath  #"D:/FirstAlignment/2024-09-19/4.jpg"
            # 获取json路径
            directory, filename_with_extension = os.path.split(template_path)
            filename, extension = os.path.splitext(filename_with_extension)
            new_extension = ".json"
            new_path = os.path.join(directory, filename + new_extension)
            file_path = new_path  #"D:/FirstAlignment/2024-09-19/0.json"
            logger.info(f'一次对准路径:')
            logger.info(input_path)
            logger.info(f'模板路径:')
            logger.info(template_path)
            logger.info(f'json路径:')
            logger.info(new_path)
            logger.info('开始获取云台参数...')
            areas = []
            # 获取俯仰角与放大倍率
            '''
            try:
                client = PTZClient()
                bearing_val = client.get_bearing_val()
                pitching_val = client.get_pitching_val()
                zoom_val = client.get_zoom()
                magnification = zoom_val
            except:
                logger.info('获取云台失败，请检查云台连接是否正确...')
            if float(bearing_val)+float(pitching_val)+float(magnification) == 0:
                time.sleep(0.2)
                try:
                    client = PTZClient()
                    bearing_val = client.get_bearing_val()
                    pitching_val = client.get_pitching_val()
                    zoom_val = client.get_zoom()
                    magnification = zoom_val
                except:
                    logger.info('获取云台失败，请检查云台连接是否正确...')
            if float(bearing_val)+float(pitching_val)+float(magnification) == 0:
                time.sleep(0.2)
                try:
                    client = PTZClient()
                    bearing_val = client.get_bearing_val()
                    pitching_val = client.get_pitching_val()
                    zoom_val = client.get_zoom()
                    magnification = zoom_val
                except:
                    logger.info('获取云台失败，请检查云台连接是否正确...')
            if float(bearing_val)+float(pitching_val)+float(magnification) == 0:
                time.sleep(0.2)
                try:
                    client = PTZClient()
                    bearing_val = client.get_bearing_val()
                    pitching_val = client.get_pitching_val()
                    zoom_val = client.get_zoom()
                    magnification = zoom_val
                except:
                    logger.info('获取云台失败，请检查云台连接是否正确...')
            if float(bearing_val)+float(pitching_val)+float(magnification) == 0:
                time.sleep(0.2)
                try:
                    client = PTZClient()
                    bearing_val = client.get_bearing_val()
                    pitching_val = client.get_pitching_val()
                    zoom_val = client.get_zoom()
                    magnification = zoom_val
                except:
                    logger.info('获取云台失败，请检查云台连接是否正确...')
            '''
            while True:
                try:
                    client = PTZClient()
                    bearing_val = client.get_bearing_val()
                    pitching_val = client.get_pitching_val()
                    # zoom_val = client.get_zoom()
                    # magnification = zoom_val
                except:
                    logger.exception('获取云台失败，请检查云台连接是否正确...')
                if float(bearing_val) == -1 or float(pitching_val) == -1:
                    logger.warning('云台获取数据异常，重新获取...')
                    time.sleep(0.2)
                else:
                    break
           
            initHorizontalAngel = float(bearing_val / 100)
            initVerticalAngel = float(pitching_val / 100)
            logger.info('云台参数获取成功！')
            # logger.info('放大倍率:')
            # logger.info(magnification)
            logger.info('水平角:')
            logger.info(bearing_val)
            logger.info('俯仰角:')
            logger.info(pitching_val)
            logger.info('开始解析json文档...')
            # # test start 
            # magnification = 4243
            # initHorizontalAngel = 18.44
            # initVerticalAngel =5.64
            # # test over


            # 打开并读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # 创建一个列表来存储所有的区域信息，每个区域信息是一个字典
            # 获取区域信息
            areas_info = data['areas']['areas_info']
            logger.info(len(areas_info))
            for area in areas_info:
                area_roi = area['area_ROI']
                meters = []
                # 获取每个区域中的仪表信息
                meters_info = area['groups']['groups_info'][0]['meters']['meters_info']
                # 将区域和其包含的仪表信息一起存储
                areas.append({
                    'area': area_roi,
                    'meters': meters
                })
                for meter in meters_info:
                    meter_roi = meter['meter_ROI']
                    meters.append(meter_roi)
                    

            # 打印结果
            for area in areas:
                for meter in area['meters']:
                    roi_x = meter['x']
                    roi_y = meter['y']
                    roi_w = meter['w']
                    roi_h = meter['h']
                    x1, y1, x2, y2 = int(roi_x), int(roi_y), int(roi_x)+int(roi_w), int(roi_y)+int(roi_h)
                    img_template = cv2.imread(template_path)
                   #  roi_template = img_template[y1:y2, x1:x2]
                    
                    # logger.info(roi_x,roi_y,roi_w,roi_h)
                    
            logger.info('开始进行第二次对准...')
            try:
                img = cv2.imread(input_path)
            except Exception as e:
                logger.exception(f"发生错误: {e}")
                _json_file_path = 'first_x_value.json'
                # /home/zngd613/robot/zk/one_align/first_x_value.json
                # 删除文件
                try:
                    os.remove(_json_file_path)
                    logger.info(f"文件 {_json_file_path} 已删除")
                except FileNotFoundError:
                    logger.error(f"文件 {_json_file_path} 不存在")
                except PermissionError:
                    logger.error(f"没有权限删除文件 {_json_file_path}")
                except Exception as e:
                    logger.exception(f"删除文件 {_json_file_path} 时发生错误: {e}")
                return DeviceIdentifyGPRCService_pb2.identifyValue(mValue=0, valueType=0)
            # img = cv2.imread(input_path)
            cv2.imwrite(f'./{dir_temp_img}/{filename}_seconed_image.jpg', img)
            img_template = unevenLightCompensate(img_template,20)
            roi_template = img_template[y1:y2, x1:x2]
            img = unevenLightCompensate(img,20)
            res = cv2.matchTemplate(img, roi_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # 绘制矩形框
            top_left = max_loc
            bottom_right = (top_left[0] + roi_w, top_left[1] + roi_h)
            # 模版匹配后的仪表图像
            roi_img = img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]  
            cv2.imwrite(f'./{dir_temp_img}/{filename}_saved_image_path2.jpg', roi_img)

            firstHorizontalPixelDifference = abs(round(first_x_value - top_left[0]))
            firstVerticalPixelDifference = abs(round(first_y_value - top_left[1]))
            
            if firstHorizontalPixelDifference==0:
                logger.info('没有检测到x轴像素变化量')
                HorizontalMagnification_k = 0
            else:
                logger.info('x轴像素变化量为:')
                logger.info(firstHorizontalPixelDifference)
                HorizontalMagnification_k = 1 / firstHorizontalPixelDifference
                logger.info('x轴像素-云台变化率为:')
                logger.info(HorizontalMagnification_k)
            
            if firstVerticalPixelDifference==0:
                logger.info('没有检测到y轴像素变化量')
                VerticalMagnification_k = 0
            else:
                logger.info('y轴像素变化量为:')
                logger.info(firstVerticalPixelDifference)
                VerticalMagnification_k = 1 / firstVerticalPixelDifference
                logger.info('y轴像素-云台变化率为:')
                logger.info(VerticalMagnification_k)
            imgCenter_x = top_left[0] + (roi_w / 2)
            imgCenter_y = top_left[1] + (roi_h / 2 )
            logger.info('图像中心点为:')
            logger.info(f'{imgCenter_x},{imgCenter_y}')
            horizontalPixelDifference =  -round(imgCenter_x - 960)
            verticalPixelDifference = -round(imgCenter_y - 540)
            logger.info('水平和垂直变化像素为:')
            logger.info(f'{horizontalPixelDifference},{verticalPixelDifference}')

            
            # 计算水平
            horizontalAngleChange = HorizontalMagnification_k * horizontalPixelDifference
            horizontalAngleFinall = initHorizontalAngel + horizontalAngleChange
            if horizontalAngleFinall < 0:
                horizontalAngleFinall = 360 + horizontalAngleFinall
            if horizontalAngleFinall > 360:
                horizontalAngleFinall = horizontalAngleFinall - 360
            # 计算俯仰
            verticalAngleChange = VerticalMagnification_k * verticalPixelDifference
            verticalAngleFinall = initVerticalAngel + verticalAngleChange
            if verticalAngleFinall < 0:
                verticalAngleFinall = 360 + verticalAngleFinall
            if verticalAngleFinall > 360:
                verticalAngleFinall = verticalAngleFinall - 360
                
            horizontalAngleFinall = int(round(horizontalAngleFinall * 100, 0))
            verticalAngleFinall = int(round(verticalAngleFinall * 100, 0))
            logger.info('第二次对准完成，云台调整到：')
            logger.info(f'{horizontalAngleFinall},{verticalAngleFinall}')
            #删除文件

            # 文件路径
            _json_file_path = 'first_x_value.json'
            # /home/zngd613/robot/zk/one_align/first_x_value.json
            # 删除文件
            try:
                os.remove(_json_file_path)
                logger.info(f"文件 {_json_file_path} 已删除")
            except FileNotFoundError:
                logger.error(f"文件 {_json_file_path} 不存在")
            except PermissionError:
                logger.error(f"没有权限删除文件 {_json_file_path}")
            except Exception as e:
                logger.exception(f"删除文件 {_json_file_path} 时发生错误: {e}")
            
        return DeviceIdentifyGPRCService_pb2.identifyValue(mValue=f"{horizontalAngleFinall}", valueType=verticalAngleFinall)

    # def Location(self, request, context):
    #     # 示例实现，返回模拟数据
    #     return DeviceIdentifyGPRCService_pb2.locationValue(x=100, y=200, w=300, h=400)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    DeviceIdentifyGPRCService_pb2_grpc.add_DeviceIdentifyGPRCServiceServicer_to_server(DeviceIdentifyGPRCService(), server)
    server.add_insecure_port('[::]:22215')
    server.start()
    logger.info("Server running on port 22215...")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()


# _json_file_path = '/home/zngd613/robot/zk/one_align/first_x_value.json'

# if os.path.exists(_json_file_path):
#     try:
#         os.remove(_json_file_path)
#         logger.info(f"文件 {_json_file_path} 已删除")
#     except FileNotFoundError:
#         logger.info(f"文件 {_json_file_path} 不存在")
#     except PermissionError:
#         logger.info(f"没有权限删除文件 {_json_file_path}")
#     except Exception as e:
#         logger.info(f"删除文件 {_json_file_path} 时发生错误: {e}")
# else:
#     pass
