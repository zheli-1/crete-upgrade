cmake_minimum_required(VERSION 2.8.7)
project(stp)

include(ExternalProject)

ExternalProject_Add(
	#stp-2.1.2
	stp-2.3.3

        GIT_REPOSITORY https://github.com/stp/stp.git
        #GIT_TAG "2.1.2"
		GIT_TAG "2.3.3"

        CMAKE_ARGS -DBUILD_SHARED_LIBS:BOOL=OFF -DENABLE_PYTHON_INTERFACE:BOOL=OFF -DBoost_USE_STATIC_LIBS:BOOL=ON -DMinisat_USE_STATIC_LIBS:BOOL=ON

	UPDATE_COMMAND ""
	INSTALL_COMMAND ""
	)