cmake_minimum_required(VERSION 2.8.7)

project(boost)

include(ExternalProject)

ExternalProject_Add(
	boost

	SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/boost-prefix/src/boost_1_71_0
	BUILD_IN_SOURCE 1

    DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}
	URL ${CMAKE_CURRENT_SOURCE_DIR}/boost_1_71_0.tar.bz2
	URL_MD5 4cdf9b5c2dc01fb2b7b733d5af30e558
	UPDATE_COMMAND ""

	CONFIGURE_COMMAND
        ./bootstrap.sh --with-toolset=clang --with-libraries=atomic,filesystem,log,program_options,random,serialization,system,test,thread,chrono,date_time

	BUILD_COMMAND ./b2 -a toolset=clang cxxflags="-std=c++11" -j7

	# > build.log 2>&1 include="-I/lib/llvm-9/include/c++/v1" linkflags="-L/lib/llvm-9/lib -Wl,-rpath,/lib/llvm-9/lib -lc++"
	#--with-toolset=clang visibility=global define=BOOST_ALL_DYN_LINK=1 define=_GLIBCXX_USE_CXX11_ABI=0 define=BOOST_NO_CXX11_HDR_CHRONO define=BOOST_NO_CXX11_HDR_MUTEX define=BOOST_SYSTEM_ENABLE_DEPRECATED linkflags="-static-libstdc++"

	INSTALL_COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/boost-prefix/src/boost_1_71_0/stage/lib ${CMAKE_BINARY_DIR}/bin/boost
	)
