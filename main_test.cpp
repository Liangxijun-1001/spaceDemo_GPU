#pragma warning(disable:4996)

#include "spacecode2d.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <set>
#include <fstream>
#include <sstream>
#include <vector>
#include "timer.h"

using namespace std;
//using namespace SpaceCode2DSystem;
using namespace BASE_XYCode;

#define NUM 10000000


void readFromFile(vector<LBPoint> &coorlist) {
	std::ifstream file("test_data.txt");
	std::string line;
	//std::vector<LBPoint> coorlist;

	if (file.is_open()) {
		while(std::getline(file, line)) {
			std::istringstream iss(line);
			std::string token;
			LBPoint geopoint;

			while (std::getline(iss, token, ','))
			{
				if (!token.empty()) {
					if (geopoint.lon == 0.0) {
						geopoint.lon = std::stod(token);
					} else {
						geopoint.lat = std::stod(token);
						coorlist.push_back(geopoint);
						geopoint.lon = 0.0;
					}
				}
			}

		}

		file.close();

		 // 打印读取的坐标
        // for (const auto& coord : coorlist) {
        //     std::cout << "Longitude: " << std::fixed << std::setprecision(7) << coord.lon << ", Latitude: " << coord.lat << std::endl;
        // }

	} else {
		std::cerr << "Unable to open file." << std::endl;
        return ;
	}
}

int main()
{
	uint64 code = 0;
	LBPoint geopoint = { 113.743515015, 34.249839290 };
    LBGridCoor coor = {0};
    LBRange range;
  	t.time();
    //LBPoint *h_geopoint = (LBPoint *)malloc(NUM * sizeof(LBPoint));
    vector<LBPoint> coorlist;
    // for (int i = 0; i < NUM; ++i)
    // {
    //     LBPoint h_geopoint;
    //     h_geopoint.lon = geopoint.lon;
    //     h_geopoint.lat = geopoint.lat;

    //     coorlist.push_back(h_geopoint);
    // }

	//readFromFile(coorlist);

//    /

        //经纬度坐标的编码示例
	//输入：geopoint经纬度坐标
	//     level,网格编码层级，范围[0,31]
	// code = toLBCode_from_LBPoint(geopoint, 31);
	// printf("%lld\n", code);

	//        LBPoint *h_geopoint = (LBPoint *)malloc(NUM * sizeof(LBPoint));
    // for(int i = 0; i < NUM; ++i)
    // {
    //     h_geopoint[i].lon = geopoint.lon;
    //     h_geopoint[i].lat = geopoint.lat;
	// 	//printf("h_geopoint[%d].lon:%7f\n", i, h_geopoint[i].lon);
	// 	//printf("h_geopoint[%d].lat:%7f\n", i, h_geopoint[i].lat);
    // }
	 t.time("copy data");

	// //计算网格编码所在层级示例
	// int level = getLevel_ofLBCode(code);
	// printf("%d\n", level);


	// //解码示例
	// geopoint = toLBPoint_from_CenterOfLBCode(code);
	// printf("%f, %f\n", geopoint.lon, geopoint.lat);

	// //编码关系/网格空间关系（-1相离、0相邻、1包含、2被包含、3相等）判断示例
	// uint64 code1 = toLBCode_from_LBPoint(geopoint, 31);
	// uint64 code2 = toLBCode_from_LBPoint(geopoint, 27);
	// int relation = getRealationshipOfTwoCodes(code1, code2);
	// printf("%d\n", relation);

	// //多边形网格化示例
	// vector<LBPoint> coorlist;
	LBPoint pointa;
	pointa = { -170.4979925, 10.8107962 }; coorlist.push_back(pointa);
	pointa = { -170.6644021, 12.1363918 }; coorlist.push_back(pointa);
	pointa = { -169.0074075, 12.3444038 }; coorlist.push_back(pointa);
	pointa = { -168.8409979, 11.0188082 }; coorlist.push_back(pointa);

	int levelG, num1, num2, num3;
	getLevelInfoOfPolygon(coorlist, 3, levelG, num1, num2, num3);

	vector<uint64> codelist;
printf("coorlist size is: %d\n", coorlist.size());
	codelist = getCodesOfPlygon_fixedlevel(coorlist, levelG, false, num2, num3);
		 t.time("getCodesOfPlygon_fixedlevel");
printf("codelist sie id : %d\n", codelist.size());
	//codelist = getCodesOfPlygon_fixedlevel(h_geopoint, 17,true);
	printf("======================\n");
	for (int i = 0; i < codelist.size(); i++) {
		cout << codelist[i] << endl;
	}
	printf("======================\n");

	return 0;
}
