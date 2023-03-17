cmake_minimum_required(VERSION 2.8.7)

project(llvm-translator)

include(ExternalProject)

# Dependency: llvm-3.4 or crete-llvm is installed
# update to llvm 9



# set(CMAKE_CXX_FLAGS "-std=c++2a")
find_package(Boost REQUIRED COMPONENTS system filesystem)
# set(Boost_INCLUDE_DIR "/path/to/boost")
include_directories(${Boost_INCLUDE_DIRS})


execute_process(COMMAND llvm-config-9 --prefix
  OUTPUT_VARIABLE LLVM_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

ExternalProject_Add(
	llvm-translator-qemu-3.0

	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm-translator-qemu-3.0
	BUILD_IN_SOURCE 1

	DOWNLOAD_COMMAND ""
	UPDATE_COMMAND ""

	CONFIGURE_COMMAND
        ./configure
        --target-list=i386-softmmu,x86_64-softmmu
        --enable-llvm
        --enable-tcg-interpreter
        --disable-zlib-test
        --disable-smartcard-nss
        --with-llvm=${LLVM_PATH}
        --host-cc=clang-9
        --cc=clang-9
        --cxx=clang++-9
        "--extra-cxxflags=-DTCG_LLVM_OFFLINE -DUSE_LLVM_9"
        # "--extra-ldflags=-L${CMAKE_BINARY_DIR}/bin -L${CMAKE_BINARY_DIR}/bin/boost -L/lib/llvm-9/lib -Wl,-rpath,/lib/llvm-9/lib -lc++"
        # "--extra-cflags=-g -mno-sse3 -I${CMAKE_SOURCE_DIR}/lib/include -I${CMAKE_BINARY_DIR}/lib/boost/boost-prefix/src/boost_1_71_0 -I/lib/llvm-9/include/c++/v1 -DTCG_LLVM_OFFLINE "
        #"--extra-cxxflags=-std=c++11 -O2 -DTCG_LLVM_OFFLINE -DUSE_LLVM_9"

        "--extra-ldflags=-L${CMAKE_BINARY_DIR}/bin"
        # "--extra-cflags=-g -mno-sse3 -I${CMAKE_SOURCE_DIR}/lib/include -DTCG_LLVM_OFFLINE"
        # "--extra-cxxflags=-ferror-limit=100 -std=c++11 -stdlib=libc++ -O2 -DTCG_LLVM_OFFLINE -DUSE_LLVM_9"
        # "--extra-cxxflags=-ferror-limit=100 -std=c++11 -nostdinc++ -nostdlib++ -O2 -DTCG_LLVM_OFFLINE -DUSE_LLVM_9"

        #BUILD_COMMAND make -j7 -DTCG_LLVM_OFFLINE -D_GLIBCXX_USE_CXX11_ABI=0 -mno-sse3 -stdlib=libc++
        #-nostdinc++ -nostdlib++ -isystem /lib/llvm-9/include/c++/v1 -L/lib/llvm-9/lib -Wl,-rpath,/lib/llvm-9/lib -lc++
        BUILD_COMMAND make -v

	INSTALL_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/llvm-translator-qemu-3.0/crete_gen_links.sh ${CMAKE_BINARY_DIR}
	)

# target_link_libraries(llvm-translator-qemu-3.0 ${Boost_LIBRARIES})
add_dependencies(llvm-translator-qemu-3.0 boost)

add_custom_target(llvm-translator-qemu-3.0-remake ALL
  #COMMAND  make -j7
  COMMAND  make -v
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/llvm-translator-qemu-3.0
  DEPENDS llvm-translator-qemu-3.0)
