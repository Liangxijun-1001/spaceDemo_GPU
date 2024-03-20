/*!
 *  \brief     SpaceCode 2D Grid system（LonLat based on merator）
 *  \details
 *  \author    LEI
 *  \version   1.0
 *  \date      2022-09
 *  \bug
 *  \warning
 *  \note
 *  \copyright
 */

#ifndef SPACECODE2D_H
#define SPACECODE2D_H

#ifndef DLL_API
#define DLL_API __declspec(dllexport)
#endif // !DLL_API

#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
typedef unsigned long long int uint64;
typedef unsigned int uint32;


	/************************************************************************/
	/* Base defination                                                      */
	/************************************************************************/
#define  MAXLEVEL_LB  31u// max level of LonLat coordinate
/**
* @brief Initialize the coordinate system parameters
* @note  Mustbe input, defualt espg 4326(tms model), 43260(xyz model)
*/
	bool /*DLL_API*/ InputParameters(int espg = 4326);

	typedef struct  LBPoint
	{
		//  LBPoint(double _lon = 0.0, double _lat = 0.0) :
		// 	lon(_lon)
		// 	, lat(_lat) {}
		 __host__ __device__ LBPoint(double _lon = 0.0, double _lat = 0.0) :
			lon(_lon)
			, lat(_lat) {}
		double lon;
		double lat;
	}LBPoint;

	/**
* @brief Coordinate structure of LBH grid
* @note  Contains level info.
*/
	typedef struct /*DLL_API*/ LBGridCoor
	{
		uint32 coI;			//grid coordinate of longitude
		uint32 coJ;			//grid coordinate of latitude
		uint32 level;	//level of LonLat Code
	}LBGridCoor;

	typedef /*DLL_API*/	struct LBRange
	{
		//  LBRange(double _lon_min = 0.0, double _lon_max = 0.0,
		// 	double _lat_min = 0.0, double _lat_max = 0.0) :
		// 	lon_min(_lon_min), lon_max(_lon_max), lat_min(_lat_min), lat_max(_lat_max) {}

		__host__ __device__ LBRange(double _lon_min = 0.0, double _lon_max = 0.0,
			double _lat_min = 0.0, double _lat_max = 0.0) :
			lon_min(_lon_min), lon_max(_lon_max), lat_min(_lat_min), lat_max(_lat_max) {}
		/*! minimum or maximum lon&lat, unit:degree */
		double lon_min;
		double lon_max;
		double lat_min;
		double lat_max;
	}LBRange;


#define BASE_SpaceCode
#ifdef BASE_SpaceCode

	/**[1-0] Encoding algorithm
	* @brief
	* @param (lon, lat) or (x, y)
	* @param level_LB, defult value: MAXLEVEL_LB
	* @param espg, defulst value:3857/Web Mercator
	* @return spacecode(uint64)
	* @note  level_LB, level_H must be inputted
	* @update 2022.09.22
	*/
	__device__ uint64 /*DLL_API*/ toLBCode_from_LBPoint(LBPoint geopoint, int level = MAXLEVEL_LB);

	/**[1-1] to GridCoor from geopoint
	* @brief
	* @param {lon,lat}
	* @param level_LB,defult value: MAXLEVEL_LB
	* @return gridcoor={coI,coJ,level}
	* @note  coI--lon    coJ--lat
	* @update 2022.09.23
	*/
	LBGridCoor /*DLL_API*/ toLBGridCoor_from_LBPoint(LBPoint geopoint, int level = MAXLEVEL_LB);
	/**[1-1] to GridCoor from geopoint
	* @brief
	* @param LBGridCoor
	* @return spacecode(uint64)
	* @update 2022.09.24
	*/
	uint64 /*DLL_API*/ toLBCode_from_LBGridCoor(LBGridCoor coor);

	__device__ uint64 /*DLL_API*/ toLBCode_from_LBGridCoor(LBGridCoor coor);


	/**[2-0] Decoding algorithm (get center point of the grid)
	* @brief
	* @param code
	* @return LBPoint
	* @note
	* @update 2022.09.23
	*/
	LBPoint /*DLL_API*/ toLBPoint_from_CenterOfLBCode(uint64 code);

	/**[2-0] Decoding algorithm (get range)
	* @brief
	* @param code
	* @return LBRange
	* @note
	* @update 2022.09.23
	*/
	__host__ __device__ LBRange /*DLL_API*/ toLBRange_from_LBCode(uint64 code);

	/**[2-1] toLBGridCoor from LBCode
	* @brief
	* @param code
	* @return LBGridCoor
	* @note
	* @update 2022.09.23
	*/
	__host__ __device__ LBGridCoor /*DLL_API*/ toLBGridCoor_from_LBCode(uint64 code);

	//LBGridCoor /*DLL_API*/ toLBGridCoor_from_LBCode(uint64 code);

	/**[2-2] get level of LBCode
	* @brief
	* @param code
	* @return level
	* @note
	* @update 2022.09.23
	*/
	__device__ uint32 /*DLL_API*/ getLevel_ofLBCode(uint64 code);


	/**[3-1] Get Parent Code
	* @brief
	* @param code, parent level
	* @return Parent code
	* @note
	* @update 2022.09.23
	*/
	__device__ uint64 /*DLL_API*/ getParent_ofLBCode(uint64 code, uint32 plevel);
	 __device__ uint64 /*DLL_API*/ getParent_ofLBCode(uint64 code);

	/**[3-2] Get Relationship of two codes
	* @brief
	* @param code1, code2
	* @return -1 [non-adjacent] 0 [adjacent]
	* @return  1 [code1 contains code2] 2 [code2 contains code1] 3 [code1 == code2]
	* @note
	* @update 2022.09.23
	*/
	 int /*DLL_API*/ getRealationshipOfTwoCodes(uint64 code1, uint64 code2);

	#define ADVANCE_SpaceCode
	#ifdef ADVANCE_SpaceCode

	/**Gridding the polygon with fixed level
	* @brief
	* @param list of (lon,lat) Note: clockwise or anti-clockwise
	* @param level
	* @param ismultiscale, defualt = true(convert grids to multiscale grid codes)
	* @return  list of codes
	* @note
	* @update 2022.09.27
	*/
	vector<uint64> /*DLL_API*/ getCodesOfPlygon_fixedlevel(vector<LBPoint> coorlist, uint32 level, bool ismultiscale = true);


	/**Get adaptive level for gridding the polygon
	* @brief
	* @param coorlist, list of (lon,lat) Note: clockwise or anti-clockwise
	* @param depth, usually set as 3,4,5
	* @return  level, used for [getCodesOfPolygon_fixedlevel]
			   num1, maximum number of the codes output by [getCodesOfPolygon_fixedlevel]
			   num2, maximum number of the codes for the polygon border
			   num3, maximum number of the codes for one line of the polygon border
	* @note
	* @update 2024.03.15
	*/
	void  getLevelInfoOfPolygon(vector<LBPoint> coorlist, int depth, int& level, int& num1, int& num2, int& num3);

	#endif //ADVANCE_SpaceCode


#endif // BASE_SpaceCode

	/** namespace BASE_XYCode
	* @brief
	* @note
	* @update
	*/
	namespace BASE_XYCode
	{
		const uint32 NMAX = MAXLEVEL_LB;
		__host__ __device__ uint32 m_NofMc(uint64 Mc);
		//uint32 m_NofMc(uint64 Mc);
		__host__ __device__ uint64 m_IJN_toMc(uint32 I_N, uint32 J_N, uint32 N);
		__host__ __device__ void m_Mc_toIJN(uint64 Mc, uint32& I, uint32& J, uint32& N);
		void m_Mc_toIJN(uint64 Mc, uint32& I, uint32& J, uint32& N);
		__device__ uint64 m_FMcone(uint64 Mc);
		uint64 m_FMcone(uint64 Mc);
		 __device__ uint64 m_FMcNF(uint64 Mc, uint32 NF);
		uint64 m_FMcNF(uint64 Mc, uint32 NF);
		 __device__ void m_SonInterval(uint64 Mc, uint32 NS, uint64& minson, uint64& maxson);
	}



#endif
