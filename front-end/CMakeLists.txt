cmake_minimum_required(VERSION 2.8.7)

project(front-end)

include(ExternalProject)

find_package(Boost REQUIRED COMPONENTS system filesystem serialization)
include_directories(${Boost_INCLUDE_DIRS})

ExternalProject_Add(
	qemu-3.0

	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/qemu-3.0
	BUILD_IN_SOURCE 1

        DOWNLOAD_COMMAND ""
	UPDATE_COMMAND ""

	CONFIGURE_COMMAND
        ./configure
        --enable-tcg-interpreter
        --target-list=i386-softmmu,x86_64-softmmu
        # --disable-werror
        # --host-cc=clang-9
        # --cc=clang-9
        # --cxx=clang++-9

        # "--extra-cxxflags=-I${CMAKE_SOURCE_DIR}/lib/include"
        #"--extra-cflags=-I${CMAKE_SOURCE_DIR}/lib/include -I${CMAKE_BINARY_DIR}/lib/boost/boost-prefix/src/boost_1_76_0 -D_GLIBCXX_USE_CXX11_ABI=0"
        "--extra-cflags=-I${CMAKE_SOURCE_DIR}/lib/include"
        "--extra-ldflags=-L${CMAKE_BINARY_DIR}/bin"
        
        # "--extra-cxxflags=-I/home/zheli/crete-upgrade/crete-dev/lib/include"
        # "--extra-cflags=-I/home/zheli/crete-upgrade/crete-dev/lib/include"
        # "--extra-ldflags=-L/home/zheli/crete-upgrade/crete-build-2.3/bin"   
        

	BUILD_COMMAND make -j7

	INSTALL_COMMAND
        #ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/qemu-3.0/i386-softmmu/crete-qemu-3.0-system-i386 ${CMAKE_BINARY_DIR}/bin/crete-qemu-3.0-system-i386 &&
        ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/qemu-3.0/x86_64-softmmu/crete-qemu-3.0-system-x86_64 ${CMAKE_BINARY_DIR}/bin/crete-qemu-3.0-system-x86_64
	)

add_dependencies(qemu-3.0 crete_test_case boost)

add_custom_target(qemu-3.0-remake ALL
  COMMAND  make -j7
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/qemu-3.0
  DEPENDS qemu-3.0)
