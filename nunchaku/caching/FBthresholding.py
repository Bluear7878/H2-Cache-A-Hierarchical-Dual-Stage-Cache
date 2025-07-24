import math

class SmartThreshold:
    """
    역 U자(bell) 스케줄 + EMA(diff) 반비례 조절을 합친 O(1) 동적 threshold.
    call 순서
        ① thr = st.get(step)  ← forward 시작 직전에 읽음
        ② … diff 계산 …
        ③ st.update(step, diff) ← forward 끝 직후 호출
    """
    def __init__(self, total_steps:int,
                 t_min:float = 0.05, t_max:float = 0.5,
                 ema_alpha:float = 0.125, ema_gain:float = 1.0):
        self.N = max(1, total_steps)
        self.t_min = t_min
        self.t_max = t_max
        self.a = ema_alpha        # EMA 스무딩 계수
        self.g = ema_gain         # diff ↔ threshold 반비례 세기
        self.ema = 0.0            # diff EMA
        self._cache = self._bell(0)   # step=0용 초기 threshold

    # ---------- public ----------
    def get(self, step:int) -> float:
        "현재 step에서 사용할 threshold 가져오기(O(1))"
        return self._cache

    def update(self, step:int, diff:float):
        "현재 step의 diff를 반영해 다음 step threshold 계산"
        # 1) EMA 갱신
        self.ema += self.a * (diff - self.ema)
        # 2) 역 U자 cap
        cap = self._bell(step+1)  # 다음 step 모양
        # 3) EMA 반비례
        reactive = self.t_max - self.g * self.ema
        reactive = max(self.t_min, reactive)
        # 4) 더 엄격한 쪽 선택
        self._cache = max(self.t_min*0.5, min(cap, reactive))

    # ---------- helpers ----------
    def _bell(self, i:int) -> float:
        x = (i + 0.5)/self.N - 0.5            #  -0.5 ~ 0.5
        w = max(0.0, 1.0 - 4.0 * x * x)       # 1‑4x²  (역 U)
        return self.t_min + (self.t_max-self.t_min) * w