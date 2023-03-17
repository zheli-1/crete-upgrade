cmake_minimum_required(VERSION 2.8.7)

project(back-end)

add_subdirectory(manager)
add_subdirectory(llvm-translator)

include(ExternalProject)

#upgraded zl3
set(KLEE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_SOURCE_DIR}/lib/include/ -I${CMAKE_BINARY_DIR}/lib/boost/boost-prefix/src/boost_1_71_0")
set(KLEE_C_FLAGS "${CMAKE_C_FLAGS} -I${CMAKE_SOURCE_DIR}/lib/include/ -I${CMAKE_BINARY_DIR}/lib/boost/boost-prefix/src/boost_1_71_0")

#set(KLEE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_SOURCE_DIR}/lib/include/ -I${CMAKE_BINARY_DIR}/lib/boost/boost-prefix/src/boost_1_71_0 -DCRETE_CONFIG=1")
#set(KLEE_C_FLAGS "${CMAKE_C_FLAGS} -I${CMAKE_SOURCE_DIR}/lib/include/ -I${CMAKE_BINARY_DIR}/lib/boost/boost-prefix/src/boost_1_71_0 -DCRETE_CONFIG=1")
set(KLEE_LD_FLAGS "-L${CMAKE_BINARY_DIR}/bin -L${CMAKE_BINARY_DIR}/bin/boost")


execute_process(COMMAND which llvm-config-9
  OUTPUT_VARIABLE LLVMCONFIG_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

ExternalProject_Add(

        klee-2.1

	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/klee-2.1
	BUILD_IN_SOURCE 0

        DOWNLOAD_COMMAND ""
	UPDATE_COMMAND ""

	CMAKE_ARGS
        -DLLVM_CONFIG_BINARY=${LLVMCONFIG_PATH}
        #-DCMAKE_CXX_COMPILER=clang++
        #-DCMAKE_C_COMPILER=clang
        -DCMAKE_CXX_FLAGS=${KLEE_CXX_FLAGS}
        #-DKLEE_COMPONENT_CXX_FLAGS=${KLEE_CXX_FLAGS}
        -DCMAKE_C_FLAGS=${KLEE_C_FLAGS}
        #-DKLEE_COMPONENT_EXTRA_INCLUDE_DIRS=${KLEE_C_FLAGS}
        -DCMAKE_EXE_LINKER_FLAGS=${KLEE_LD_FLAGS}
        #-DKLEE_COMPONENT_EXTRA_LIBRARIES=${KLEE_LD_FLAGS}
        #-DBUILD_SHARED_LIBS:BOOL=OFF -DENABLE_PYTHON_INTERFACE:BOOL=OFF -DBoost_USE_STATIC_LIBS:BOOL=ON -DMinisat_USE_STATIC_LIBS:BOOL=ON
        -DENABLE_SOLVER_STP=ON
        -DENABLE_SOLVER_Z3=OFF
        -DENABLE_SYSTEM_TESTS=OFF
        -DENABLE_UNIT_TESTS=OFF 
        -DENABLE_KLEE_ASSERTS=FALSE
        -DENABLE_TCMALLOC=OFF 

        INSTALL_COMMAND
        ln -sf

        ${CMAKE_CURRENT_BINARY_DIR}/klee-2.1-prefix/src/klee-2.1-build/bin/klee
        ${CMAKE_BINARY_DIR}/bin/crete-klee-2.1

        BUILD_ALWAYS 1
	)
add_dependencies(klee-2.1 stp-2.3.3 crete_test_case boost)
