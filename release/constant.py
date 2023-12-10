# from django.templatetags.static import static
import os


def constant(func):
    def func_set(self, value):
        raise TypeError
    
    def func_get(self):
        return func()
    return property(func_get, func_set)

class _Const(object):
    # @constant
    # def CURRENT_PATH():
    #     return static('core/')

    @constant
    def CURRENT_PATH():
        return os.path.dirname(os.path.abspath(__file__)) + '/data'
    
    @constant
    def ACCX():
        return 'AccelerationX'
    
    @constant
    def ACCY():
        return 'AccelerationY'
    
    @constant
    def ACCZ():
        return 'AccelerationZ'
    
    @constant
    def RVP():
        return 'RRX'
    
    @constant
    def RVR():
        return 'RRY'
    
    @constant
    def RVY():
        return 'RRZ'
    
    @constant
    def PITCH():
        return 'Pitch'
    
    @constant
    def ROLL():
        return 'Roll'
    
    @constant
    def YAW():
        return 'Yaw'
    
    @constant
    def TIMESTAMP():
        return 'Timestamp'
    
    @constant
    def FEATURE():
        return 'Feature'
    
    @constant
    def VALUE():
        return 'Value'
    
    # for Analysis
    @constant
    def SWING_WINDOW_SIZE():
        return 8
    
    @constant
    def SWING_BEFORE_IMPACT():
        return 10
    
    @constant
    def SWING_AFTER_IMPACT():
        return 5
        
    @constant
    def SWING_PREPARE_END():
        return 6
    
    @constant
    def SWING_IMPACT_END():
        return 11
    
    @constant
    def SWING_INTERPRET_RES():
        return {'000': '준비 스윙 속도 부족',
                '001': '준비 스윙 속도 부족',
                '010': '준비 스윙 궤적 이상',
                '020': '010 준비 스윙 손목 방향 잘못됨',
                '021': '010 준비 스윙 손목 방향 잘못됨',
                '030': '010 준비 스윙 깊이 부족',
                '031': '010 준비 스윙 깊이 부족',
                '100': '임펙트 스윙 속도 부족',
                '101': '임펙트 스윙 속도 부족',
                '110': '임펙트 스윙 손목 방향 잘못됨',
                '120': '110 임펙트 스윙 손목 회전 미사용',
                '130': '110 임펙트 스윙 손목 회전 이상',
                '131': '110 임펙트 스윙 손목 회전 이상',
                '200': '임펙트 이후 자세가 부자연스러움',
                '201': '임펙트 이후 자세가 부자연스러움',
                '210': '손목회전이 민첩하지 않음',
                '211': '손목회전이 민첩하지 않음',
                '220': '팔로우 스윙 과정에서 손목이 꺾임',
                '221': '팔로우 스윙 과정에서 손목이 꺾임',
                '500': '백스윙 속도 적절',
                '510': '백스윙 궤적 적절',
                '600': '임펙트 스윙 속도 적절',
                '610': '임펙트 스윙 시 손목을 올바르게 회전',
                '700': '임펙트 이후 자세가 안정적',
                '710': '임펙트 이후 손목회전이 자연스러움',
                '720': '임펙트 이후 손목이 꺾이지 않았음',
                }
    

    # for Classification
    @constant
    def CLASS_STROKE():
        return ['bd', 'bn', 'bh', 'bu', 'fd', 'fp', 'fn', 'fh', 'fs', 'fu', 'ls', 'ss']
    
    @constant
    def CLASS_START():
        return 'StartFrame'
    
    @constant
    def CLASS_END():
        return 'EndFrame'
    
    @constant
    def CLASS_WINDOW_SIZE():
        return 30
    
    @constant
    def CLASS_ACC_THR():
        return 2
    
    @constant
    def CLASS_RR_THR():
        return 200
    
    @constant
    def CLASS_FEAT_THR():
        return 1
    
    @constant
    def CLASS_PEAK_THR():
        return 3
    
    @constant
    def CLASS_COLS():
        return [CONST.ACCX, CONST.ACCY, CONST.ACCZ, CONST.RVP, CONST.RVR, CONST.RVY]
    
CONST = _Const() if not hasattr(_Const, '_instance') else _Const._instance