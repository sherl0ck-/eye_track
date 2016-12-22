#find_library(MGL mgl PATHS /usr/local/lib)
ADD_EXECUTABLE(eyeLike main.cpp findEyeCenter.cpp findEyeCenter.h constants.h helpers.cpp helpers.h haar.h haar.cpp maps.h maps.cpp detect.h detect.cpp slic.h slic.cpp saliency.h saliency.cpp)
TARGET_LINK_LIBRARIES( eyeLike ${OpenCV_LIBS} )
#target_link_libraries( eyeLike ${MGL})