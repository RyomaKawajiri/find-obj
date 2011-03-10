CXXFLAGS=`pkg-config --cflags opencv` -O3 -Wall
LDFLAGS=`pkg-config --libs opencv`

EXE=find_obj
OBJS=find_obj.o
all: ${EXE}

find_obj: ${OBJS}
	${CXX} ${LDFLAGS} ${OBJS} -o $@

clean:
	rm -rf ${EXE} ${OBJS} *~
