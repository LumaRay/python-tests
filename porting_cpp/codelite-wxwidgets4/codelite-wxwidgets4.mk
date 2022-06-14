##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Debug
ProjectName            :=codelite-wxwidgets4
ConfigurationName      :=Debug
WorkspaceConfiguration := $(ConfigurationName)
WorkspacePath          :=/home/thermalview/Desktop/ThermalView/tests/porting_cpp
ProjectPath            :=/home/thermalview/Desktop/ThermalView/tests/porting_cpp/codelite-wxwidgets4
IntermediateDirectory  :=../build-$(ConfigurationName)/codelite-wxwidgets4
OutDir                 :=../build-$(ConfigurationName)/codelite-wxwidgets4
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=ThermalView
Date                   :=02/03/22
CodeLitePath           :=/home/thermalview/.codelite
LinkerName             :=/usr/bin/g++-6
SharedObjectLinkerName :=/usr/bin/g++-6 -shared -fPIC
ObjectSuffix           :=.o
DependSuffix           :=.o.d
PreprocessSuffix       :=.i
DebugSwitch            :=-g 
IncludeSwitch          :=-I
LibrarySwitch          :=-l
OutputSwitch           :=-o 
LibraryPathSwitch      :=-L
PreprocessorSwitch     :=-D
SourceSwitch           :=-c 
OutputFile             :=../build-$(ConfigurationName)/bin/$(ProjectName)
Preprocessors          :=
ObjectSwitch           :=-o 
ArchiveOutputSwitch    := 
PreprocessOnlySwitch   :=-E
ObjectsFileList        :=$(IntermediateDirectory)/ObjectsList.txt
PCHCompileFlags        :=
LinkOptions            :=  -fopenmp $(shell wx-config --libs)
IncludePath            :=  $(IncludeSwitch). $(IncludeSwitch). $(IncludeSwitch)/home/thermalview/.local/include/opencv4/ $(IncludeSwitch)/usr/local/include/torch/csrc/api/include/ 
IncludePCH             := 
RcIncludePath          := 
Libs                   := $(LibrarySwitch)wx_baseu-3.1 $(LibrarySwitch)wx_gtk3u_stc-3.1 $(LibrarySwitch)wx_baseu_net-3.1 $(LibrarySwitch)wx_gtk3u_media-3.1 $(LibrarySwitch)wx_gtk3u_aui-3.1 $(LibrarySwitch)wx_gtk3u_ribbon-3.1 $(LibrarySwitch)wx_gtk3u_html-3.1 $(LibrarySwitch)wx_gtk3u_richtext-3.1 $(LibrarySwitch)wx_baseu_xml-3.1 $(LibrarySwitch)wx_gtk3u_core-3.1 $(LibrarySwitch)wx_gtk3u_adv-3.1 $(LibrarySwitch)wx_gtk3u_xrc-3.1 $(LibrarySwitch)wx_gtk3u_propgrid-3.1 $(LibrarySwitch)wx_gtk3u_gl-3.1 $(LibrarySwitch)opencv_core $(LibrarySwitch)opencv_imgproc $(LibrarySwitch)opencv_imgcodecs $(LibrarySwitch)opencv_highgui $(LibrarySwitch)shm $(LibrarySwitch)nvrtc $(LibrarySwitch)cudnn $(LibrarySwitch)cuda $(LibrarySwitch)cudart $(LibrarySwitch)cublas $(LibrarySwitch)curand $(LibrarySwitch)cusolver $(LibrarySwitch)nvrtc-builtins $(LibrarySwitch)nvToolsExt $(LibrarySwitch)caffe2_observers $(LibrarySwitch)caffe2_nvrtc $(LibrarySwitch)caffe2_module_test_dynamic $(LibrarySwitch)caffe2_detectron_ops_gpu $(LibrarySwitch)torch $(LibrarySwitch)torch_global_deps $(LibrarySwitch)torch_cuda $(LibrarySwitch)c10_cuda $(LibrarySwitch)c10 $(LibrarySwitch)torch_cpu 
ArLibs                 :=  "libwx_baseu-3.1.so" "libwx_gtk3u_stc-3.1.so" "libwx_baseu_net-3.1.so" "libwx_gtk3u_media-3.1.so" "libwx_gtk3u_aui-3.1.so" "libwx_gtk3u_ribbon-3.1.so" "libwx_gtk3u_html-3.1.so" "libwx_gtk3u_richtext-3.1.so" "libwx_baseu_xml-3.1.so" "libwx_gtk3u_core-3.1.so" "libwx_gtk3u_adv-3.1.so" "libwx_gtk3u_xrc-3.1.so" "libwx_gtk3u_propgrid-3.1.so" "libwx_gtk3u_gl-3.1.so" "libopencv_core.so" "libopencv_imgproc.so" "libopencv_imgcodecs.so" "libopencv_highgui.so" "libshm.so" "libnvrtc.so" "libcudnn.so" "libcuda.so" "libcudart.so" "libcublas.so" "libcurand.so" "libcusolver.so" "libnvrtc-builtins.so" "libnvToolsExt.so" "libcaffe2_observers.so" "libcaffe2_nvrtc.so" "libcaffe2_module_test_dynamic.so" "libcaffe2_detectron_ops_gpu.so" "libtorch.so" "libtorch_global_deps.so" "libtorch_cuda.so" "libc10_cuda.so" "libc10.so" "libtorch_cpu.so" 
LibPath                := $(LibraryPathSwitch). $(LibraryPathSwitch)/home/thermalview/Phoenix/build/wxbld/gtk3/lib/ $(LibraryPathSwitch)/usr/local/cuda-10.1/targets/x86_64-linux/lib/ $(LibraryPathSwitch)/home/thermalview/.local/lib/ $(LibraryPathSwitch)/usr/local/lib/python3.6/dist-packages/torch/lib/ 

##
## Common variables
## AR, CXX, CC, AS, CXXFLAGS and CFLAGS can be overriden using an environment variables
##
AR       := /usr/bin/ar rcu
CXX      := /usr/bin/g++-6
CC       := /usr/bin/gcc-6
CXXFLAGS :=  -g -O0 -fopenmp -Wall $(shell wx-config --cflags) $(Preprocessors)
CFLAGS   :=  -g -O0 -Wall $(Preprocessors)
ASFLAGS  := 
AS       := /usr/bin/as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Objects0=../build-$(ConfigurationName)/codelite-wxwidgets4/test_torch.cpp$(ObjectSuffix) ../build-$(ConfigurationName)/codelite-wxwidgets4/test_opencv.cpp$(ObjectSuffix) ../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter_bitmaps.cpp$(ObjectSuffix) ../build-$(ConfigurationName)/codelite-wxwidgets4/main.cpp$(ObjectSuffix) ../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter.cpp$(ObjectSuffix) ../build-$(ConfigurationName)/codelite-wxwidgets4/MainFrame.cpp$(ObjectSuffix) 



Objects=$(Objects0) 

##
## Main Build Targets 
##
.PHONY: all clean PreBuild PrePreBuild PostBuild MakeIntermediateDirs
all: MakeIntermediateDirs $(OutputFile)

$(OutputFile): ../build-$(ConfigurationName)/codelite-wxwidgets4/.d $(Objects) 
	@mkdir -p "../build-$(ConfigurationName)/codelite-wxwidgets4"
	@echo "" > $(IntermediateDirectory)/.d
	@echo $(Objects0)  > $(ObjectsFileList)
	$(LinkerName) $(OutputSwitch)$(OutputFile) @$(ObjectsFileList) $(LibPath) $(Libs) $(LinkOptions)

MakeIntermediateDirs:
	@mkdir -p "../build-$(ConfigurationName)/codelite-wxwidgets4"
	@mkdir -p ""../build-$(ConfigurationName)/bin""

../build-$(ConfigurationName)/codelite-wxwidgets4/.d:
	@mkdir -p "../build-$(ConfigurationName)/codelite-wxwidgets4"

PreBuild:


##
## Objects
##
../build-$(ConfigurationName)/codelite-wxwidgets4/test_torch.cpp$(ObjectSuffix): test_torch.cpp ../build-$(ConfigurationName)/codelite-wxwidgets4/test_torch.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/thermalview/Desktop/ThermalView/tests/porting_cpp/codelite-wxwidgets4/test_torch.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/test_torch.cpp$(ObjectSuffix) $(IncludePath)
../build-$(ConfigurationName)/codelite-wxwidgets4/test_torch.cpp$(DependSuffix): test_torch.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT../build-$(ConfigurationName)/codelite-wxwidgets4/test_torch.cpp$(ObjectSuffix) -MF../build-$(ConfigurationName)/codelite-wxwidgets4/test_torch.cpp$(DependSuffix) -MM test_torch.cpp

../build-$(ConfigurationName)/codelite-wxwidgets4/test_torch.cpp$(PreprocessSuffix): test_torch.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) ../build-$(ConfigurationName)/codelite-wxwidgets4/test_torch.cpp$(PreprocessSuffix) test_torch.cpp

../build-$(ConfigurationName)/codelite-wxwidgets4/test_opencv.cpp$(ObjectSuffix): test_opencv.cpp ../build-$(ConfigurationName)/codelite-wxwidgets4/test_opencv.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/thermalview/Desktop/ThermalView/tests/porting_cpp/codelite-wxwidgets4/test_opencv.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/test_opencv.cpp$(ObjectSuffix) $(IncludePath)
../build-$(ConfigurationName)/codelite-wxwidgets4/test_opencv.cpp$(DependSuffix): test_opencv.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT../build-$(ConfigurationName)/codelite-wxwidgets4/test_opencv.cpp$(ObjectSuffix) -MF../build-$(ConfigurationName)/codelite-wxwidgets4/test_opencv.cpp$(DependSuffix) -MM test_opencv.cpp

../build-$(ConfigurationName)/codelite-wxwidgets4/test_opencv.cpp$(PreprocessSuffix): test_opencv.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) ../build-$(ConfigurationName)/codelite-wxwidgets4/test_opencv.cpp$(PreprocessSuffix) test_opencv.cpp

../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter_bitmaps.cpp$(ObjectSuffix): wxcrafter_bitmaps.cpp ../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter_bitmaps.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/thermalview/Desktop/ThermalView/tests/porting_cpp/codelite-wxwidgets4/wxcrafter_bitmaps.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/wxcrafter_bitmaps.cpp$(ObjectSuffix) $(IncludePath)
../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter_bitmaps.cpp$(DependSuffix): wxcrafter_bitmaps.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter_bitmaps.cpp$(ObjectSuffix) -MF../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter_bitmaps.cpp$(DependSuffix) -MM wxcrafter_bitmaps.cpp

../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter_bitmaps.cpp$(PreprocessSuffix): wxcrafter_bitmaps.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) ../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter_bitmaps.cpp$(PreprocessSuffix) wxcrafter_bitmaps.cpp

../build-$(ConfigurationName)/codelite-wxwidgets4/main.cpp$(ObjectSuffix): main.cpp ../build-$(ConfigurationName)/codelite-wxwidgets4/main.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/thermalview/Desktop/ThermalView/tests/porting_cpp/codelite-wxwidgets4/main.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IncludePath)
../build-$(ConfigurationName)/codelite-wxwidgets4/main.cpp$(DependSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT../build-$(ConfigurationName)/codelite-wxwidgets4/main.cpp$(ObjectSuffix) -MF../build-$(ConfigurationName)/codelite-wxwidgets4/main.cpp$(DependSuffix) -MM main.cpp

../build-$(ConfigurationName)/codelite-wxwidgets4/main.cpp$(PreprocessSuffix): main.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) ../build-$(ConfigurationName)/codelite-wxwidgets4/main.cpp$(PreprocessSuffix) main.cpp

../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter.cpp$(ObjectSuffix): wxcrafter.cpp ../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/thermalview/Desktop/ThermalView/tests/porting_cpp/codelite-wxwidgets4/wxcrafter.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/wxcrafter.cpp$(ObjectSuffix) $(IncludePath)
../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter.cpp$(DependSuffix): wxcrafter.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter.cpp$(ObjectSuffix) -MF../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter.cpp$(DependSuffix) -MM wxcrafter.cpp

../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter.cpp$(PreprocessSuffix): wxcrafter.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) ../build-$(ConfigurationName)/codelite-wxwidgets4/wxcrafter.cpp$(PreprocessSuffix) wxcrafter.cpp

../build-$(ConfigurationName)/codelite-wxwidgets4/MainFrame.cpp$(ObjectSuffix): MainFrame.cpp ../build-$(ConfigurationName)/codelite-wxwidgets4/MainFrame.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/thermalview/Desktop/ThermalView/tests/porting_cpp/codelite-wxwidgets4/MainFrame.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/MainFrame.cpp$(ObjectSuffix) $(IncludePath)
../build-$(ConfigurationName)/codelite-wxwidgets4/MainFrame.cpp$(DependSuffix): MainFrame.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT../build-$(ConfigurationName)/codelite-wxwidgets4/MainFrame.cpp$(ObjectSuffix) -MF../build-$(ConfigurationName)/codelite-wxwidgets4/MainFrame.cpp$(DependSuffix) -MM MainFrame.cpp

../build-$(ConfigurationName)/codelite-wxwidgets4/MainFrame.cpp$(PreprocessSuffix): MainFrame.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) ../build-$(ConfigurationName)/codelite-wxwidgets4/MainFrame.cpp$(PreprocessSuffix) MainFrame.cpp


-include ../build-$(ConfigurationName)/codelite-wxwidgets4//*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r $(IntermediateDirectory)


