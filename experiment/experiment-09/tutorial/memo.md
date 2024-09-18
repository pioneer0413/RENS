# 라벨 생성
- CropTime
- DropEventByTime

# 강건성
- Denoise: 특정 기간 동안 연속적이지 않은 이벤트는 노이즈로 간주, 삭제함
- TimeJitter: event의 timestamp를 가우시안 분포에 따라 흐트림
- UniformNoise
- SpatialJitter: event의 pixel을 가우시안 분포에 따라 흐트림
- DropPixel

# 일반화
- Downsample
- RandomFlipUD
- RandomFlipLR
- RandomCrop
- CenterCrop
- RandomTimeReversal