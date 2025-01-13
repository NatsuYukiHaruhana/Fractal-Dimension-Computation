help: 			## Show this help
	@echo -e "\nSpecify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help build_cpu build_gpu clean

build_cpu:
	cmake -Gninja -Bbuild_cpu -DCMAKE_BUILD_TYPE=Release -S. -Bbuild_cpu -C CMakeLists.txt

build_gpu:
	cmake -Gninja -Bbuild_gpu -DCMAKE_BUILD_TYPE=Release -S. -Bbuild_gpu -C CMakeLists_GPU.txt

clean:
	rm -rf build_cpu build_gpu