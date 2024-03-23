#include "spacecode2d.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <set>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;
using namespace BASE_XYCode;

#define MAX_MIDCODES 1000
#define MAX_ALLCODES 1000

#ifndef CHAR_BIT
#define CHAR_BIT __CHAR_BIT__
#endif

#ifndef min
#define min(a, b) (((a) < (b) ? (a) : (b)))
#endif

#ifndef max
#define max(a, b) (((a) > (b) ? (a) : (b)))
#endif

/*Initialize the coordinate system parameters*/
int ESPGCode = 4326; // tms model
set<int> ESPGList;
__host__ __device__ int d_ESPGCode = 4326; // 在设备端定义对应的变量

#define PI 3.1415926535898
#define LON_MIN -180.000
#define LON_MAX 180.000
#define LAT_MIN -85.05112878
#define LAT_MAX 85.05112878
const double LON_MOVE = -LON_MIN;
const double LON_ALPHA =
    (double)(1u << MAXLEVEL_LB) / double(LON_MAX - LON_MIN);
const double LAT_MOVE = -LAT_MIN;
const double LAT_ALPHA =
    (double)(1u << MAXLEVEL_LB) / double(LAT_MAX - LAT_MIN);

#define GET_CUDA_EXE_STATUS                                                           \
  cudaError_t cudaStatus = cudaGetLastError();                                        \
  if (cudaStatus != cudaSuccess)                                                      \
  {                                                                                   \
    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); \
  }

// __device__  uint64 midcodes[MAX_MIDCODES];
// __device__  uint64 allcodes[MAX_MIDCODES];
// __device__  LBGridCoor ij_list[MAX_ALLCODES];

static const double Epsilon0 = 1.0e-10;
__device__ uint64 double_to_uint64(double val)
{
  long long int temp = static_cast<long long int>(val);
  double ferror = val - static_cast<double>(temp);
  if (0.99 < fabs(ferror))
  {
    if (0 < ferror)
    {
      temp++;
    }
    else
    {
      temp--;
    }
  }
  return temp;
}

__device__ bool isOutOf_LBPointBorder(LBPoint &geopoint)
{
  LBPoint oldgeopoint = geopoint;
  bool judge = false;
  geopoint.lon = max(LON_MIN, min(geopoint.lon, LON_MAX));
  geopoint.lat = max(LAT_MIN, min(geopoint.lat, LAT_MAX));
  if ((geopoint.lon != oldgeopoint.lon) || (geopoint.lat != oldgeopoint.lat))
  {
    judge = true;
  }
  return judge;
}

bool InputParameters(int espg)
{
  set<int> espglist;
  espglist.insert(4326);  // Web Meractor(tms模式)
  espglist.insert(43260); // Web Meractor(xyz模式)
  espglist.insert(1);     // Local grid system

  if (espglist.find(espg) == espglist.end())
  {
    cout << "error, please input right ESPG code, defualt: 1 (Local System), "
            "4326 (Web Meractor System)"
         << endl;
    return false;
  }
  else
  {
    ESPGCode = espg;
    cout << "Selected ESPG Code:" << ESPGCode << endl;
    switch (ESPGCode)
    {
    case 4326:
      cout << "Web Meractor (TMS Model)" << endl;
      break;
    case 43260:
      cout << "Web Meractor (XYZ Model)" << endl;
      break;
    default:
      break;
    }
    return true;
  }
}

	void getLevelInfoOfPolygon(vector<LBPoint> coorlist, int depth, int& level, int& num1, int& num2, int& num3)
	{
		if (depth < 0) {
			depth = 0;
			printf("Error input value of the depth for gridding the polygon, it is modified as 0.\n");
		}

		int pointnum = (int)coorlist.size();
		if (pointnum < 2) {
			printf("The point num in the polygon coordinates list is less than 2.\n");
			level = 31;
			num1 = num2 = num3 = 1;
			return;
		}

		//cal max/min of lng/lat
		double lngmin = coorlist[0].lon;
		double lngmax = coorlist[0].lon;
		double latmin = coorlist[0].lat;
		double latmax = coorlist[0].lat;
		for (int i = 1; i < pointnum; i++) {
			lngmin = min(lngmin, coorlist[i].lon);
			lngmax = max(lngmax, coorlist[i].lon);
			latmin = min(latmin, coorlist[i].lat);
			latmax = max(latmax, coorlist[i].lat);
		}

		int level_lng;
		double arg1 = abs(360.0 / (lngmax - lngmin));
		level_lng = floor(log(arg1)/log(2)); //log2(x)=log(x)/log(2), ln(x)=log(x)

		int level_lat;
		double a1 = log(tan(PI * latmin / 180.0) + 1.0 / cos(PI * latmin / 180.0));
		double a2 = log(tan(PI * latmax / 180.0) + 1.0 / cos(PI * latmax / 180.0));
		double arg2 = abs(2 * PI / (a2 - a1));
		level_lat = floor(log(arg2)/log(2));

		int level0 = min(level_lng, level_lat);

		level = level0 + depth;
		level = min(MAXLEVEL_LB, max(0, level));

		depth = level - level0;

		num1 = 1 << ((depth + 1) << 1);//pow(4,(depth + 1))
		num2 = num1 + pointnum;
		num3 = 1 << (depth + 2);//pow(2,depth+1)*2

		return;
	}


__device__ LBGridCoor toLBGridCoor_from_LBPoint_ESPG4326(LBPoint geopoint,
                                                         int level)
{
  LBGridCoor gc;
  isOutOf_LBPointBorder(geopoint);
  gc.level = max(0, min(level, MAXLEVEL_LB));
  uint64 coI = double_to_uint64((geopoint.lon + LON_MOVE) * LON_ALPHA) >>
               (MAXLEVEL_LB - gc.level);

  uint32 zoom = 1ul << (MAXLEVEL_LB);
  double lat_rad = geopoint.lat * PI / 180.00;
  double tanlat_rad = tan(lat_rad);
  double seclat_rad = (1.00 / cos(lat_rad));
  double doublecoJ =
      double(zoom) * (1 - (log(tanlat_rad + seclat_rad) / PI)) * 0.50;
  uint64 coJ = double_to_uint64(doublecoJ) >> (MAXLEVEL_LB - gc.level);

  if (d_ESPGCode == 4326)
  {
    coJ = (1ull << gc.level) - 1 - coJ; // tms model
  }

  gc.coI = uint32(coI), gc.coJ = uint32(coJ);
  return gc;
}

__host__ __device__ LBRange toLBRange_from_LBGridCoor_ESPG4326(LBGridCoor coor)
{
  LBRange range = {0.0};
  coor.coI = coor.coI << (MAXLEVEL_LB - coor.level);
  range.lon_min = (double(coor.coI) / double(LON_ALPHA)) - LON_MOVE;
  double detalon =
      double(1ul << (MAXLEVEL_LB - coor.level)) / double(LON_ALPHA);
  range.lon_max = range.lon_min + detalon;

  if (d_ESPGCode == 4326)
  {
    coor.coJ = (1ull << coor.level) - 1 - coor.coJ; // tms model
  }

  double zoom = pow(2, coor.level);
  double temp = PI * (1.0 - 2.0 * double(coor.coJ + 1) / zoom);
  double lat_rad = atan(sinh(temp));
  range.lat_min = lat_rad * 180.00 / PI;

  temp = PI * (1.0 - 2.0 * double(coor.coJ) / zoom);
  lat_rad = atan(sinh(temp));
  range.lat_max = lat_rad * 180.00 / PI;

  return range;
}

//*************************************************************************************************
// 锟接空硷拷namespace LBHGridSystem::BASE_XYCode
namespace BASE_XYCode
{
  __host__ __device__ const unsigned long long int L[6] = {
      0xFFFFFFFF00000000, 0xFFFF0000, 0xFF00, 0xF0, 0xC, 0x2};
  __host__ __device__ const unsigned long long int R[6] = {
      0x00000000FFFFFFFF, 0x0000FFFF, 0x00FF, 0x0F, 0x3, 0x1};
  __host__ __device__ const int NN[6] = {32, 16, 8, 4, 2, 1};
  static unsigned long long int m_Crossbit(uint32 I, uint32 J)
  {
    uint64 Sc = 0ull;
    for (int i = 0; i < sizeof(I) * CHAR_BIT; i++)
    {
      Sc |= (I & 1ull << i) << i | (J & 1ull << i) << (i + 1);
    }
    return Sc;
  }

  static void m_CrossbitRE(uint64 Sc, uint32 &I, uint32 &J)
  {
    I = 0ul;
    J = 0ul;
    for (int temp_i = 0; temp_i < sizeof(Sc) * CHAR_BIT; temp_i++)
    {
      I |= ((Sc & (1ull << temp_i)) >> (temp_i >> 1));
      temp_i++;
      J |= ((Sc & (1ull << temp_i)) >> ((temp_i >> 1) + 1));
    }
  }

  static uint64 m_McNi(uint32 N, uint64 ni)
  {
    uint64 maxnum = 1ull << (N + N);
    if (ni > maxnum)
    {
      ni = maxnum;
    }

    if (ni == 0)
    {
      ni = 1;
    }
    ni = ni - 1;
    uint64 Mc;
    Mc = ((1ull << (NMAX + NMAX - (N << 1))) - 1) +
         (ni << (NMAX + NMAX + 1 - (N << 1)));
    return Mc;
  }

  __host__ __device__ unsigned int m_ReturnmidN(uint64 &mid, uint32 ceng)
  {
#ifdef __CUDA_ARCH__
    uint64 mid0 = (mid & L[ceng]) >> NN[ceng];
    if (mid0)
    {
      mid = mid0;
      return 0;
    }
    mid = mid & R[ceng];
    return NN[ceng];
#else
    uint64 mid0 = (mid & L[ceng]) >> NN[ceng];
    if (mid0)
    {
      mid = mid0;
      return 0;
    }
    mid = mid & R[ceng];
    return NN[ceng];
#endif
  }

  //  unsigned int m_ReturnmidN(uint64 &mid, uint32 ceng)
  // {
  //   uint64 mid0 = (mid & L[ceng]) >> NN[ceng];
  //   if (mid0)
  //   {
  //     mid = mid0;
  //     return 0;
  //   }
  //   mid = mid & R[ceng];
  //   return NN[ceng];
  // }

  __host__ __device__ uint32 m_NofMc(uint64 Mc)
  {
    if (!(Mc & 1))
    {
      return NMAX;
    }
    uint64 mid = (Mc - 1) ^ (Mc + 1);
    int N = 0;
    int i;
    for (i = 0; i < 6; i++)
    {
      N = N + m_ReturnmidN(mid, i);
    }
    unsigned int Nbian = (N >> 1) - 31 + NMAX;
    if (Nbian > NMAX)
    {
      printf("error m_NofMc \n");
      Nbian = 0;
    }
    return Nbian;
  }

  // uint32 m_NofMc(uint64 Mc)
  // {
  //   if (!(Mc & 1))
  //   {
  //     return NMAX;
  //   }
  //   uint64 mid = (Mc - 1) ^ (Mc + 1);
  //   int N = 0;
  //   int i;
  //   for (i = 0; i < 6; i++)
  //   {
  //     N = N + m_ReturnmidN(mid, i);
  //   }
  //   unsigned int Nbian = (N >> 1) - 31 + NMAX;
  //   if (Nbian > NMAX)
  //   {
  //     printf("error m_NofMc \n");
  //     Nbian = 0;
  //   }
  //   return Nbian;
  // }

  __device__ uint64 m_Crossbit_N_toMc(uint32 I_N, uint32 J_N, uint32 N)
  {

    uint32 maxIJ = (1 << N) - 1;

    if (I_N > maxIJ)
    {
      if (I_N >= 4294967295)
      {
        I_N = 0;
      }
      else
      {
        I_N = maxIJ;
      }
    }
    if (J_N > maxIJ)
    {
      if (J_N >= 4294967295)
      {
        J_N = 0;
      }
      else
      {
        J_N = maxIJ;
      }
    }

    uint32 I = I_N << (NMAX - N);
    uint32 J = J_N << (NMAX - N);
    uint64 Sc = 0ull;
    uint64 Mc = 0ull;
    for (int i = 0; i < sizeof(I) * CHAR_BIT; i++)
    {
      Sc |= (I & 1ull << i) << i | (J & 1ull << i) << (i + 1);
    }
    Mc = m_FMcNF(Sc << 1, N);
    return Mc;
  }

  __device__ unsigned long long int m_Mc31A(uint64 Mc)
  {
    unsigned long long int Mc31A;
    uint32 N = 0u;
    N = m_NofMc(Mc);
    uint64 Mc0 = 0ull;
    Mc0 = (1ull << (NMAX + NMAX - N - N)) - 1;
    Mc31A = Mc - Mc0;
    return Mc31A;
  }

  __device__ void m_CrossbitRE_Mc31_toIJ_N(uint64 Mc, uint32 N, uint32 &I_N,
                                           uint32 &J_N)
  {
    uint64 Mc31A = 0ull;
    Mc31A = m_Mc31A(Mc);

    uint64 Sc = Mc31A >> 1;
    uint32 I = 0ul;
    uint32 J = 0ul;
    for (int temp_i = 0; temp_i < sizeof(Sc) * CHAR_BIT; temp_i++)
    {
      I |= ((Sc & (1ull << temp_i)) >> (temp_i >> 1));
      temp_i++;
      J |= ((Sc & (1ull << temp_i)) >> ((temp_i >> 1) + 1));
    }

    I_N = I >> (NMAX - N);
    J_N = J >> (NMAX - N);
  }

  __host__ __device__ uint64 m_IJN_toMc(uint32 I_N, uint32 J_N, uint32 N)
  {
    if (N > NMAX)
    {
      N = NMAX;
    }

    uint32 maxIJ = (1 << N) - 1;

    I_N = max(0, min(maxIJ, I_N));
    J_N = max(0, min(maxIJ, J_N));

    uint32 I = I_N;
    uint32 J = J_N;
    uint64 Sc = 0ull;
    uint64 Mc = 0ull;
    for (uint32 i = 0; i < N; i++)
    {
      Sc |= (I & 1ull << i) << i | (J & 1ull << i) << (i + 1);
    }

    uint64 Mc0 = (1ull << (NMAX + NMAX - N - N)) - 1;
    uint64 detaMc = 0ull;
    detaMc = Sc << (NMAX + NMAX + 1 - N - N);
    Mc = Mc0 + detaMc;
    return Mc;
  }

  // uint64 m_IJN_toMc(uint32 I_N, uint32 J_N, uint32 N)
  //   {
  //     if (N > NMAX)
  //     {
  //       N = NMAX;
  //     }

  //     uint32 maxIJ = (1 << N) - 1;

  //     I_N = max(0, min(maxIJ, I_N));
  //     J_N = max(0, min(maxIJ, J_N));

  //     uint32 I = I_N;
  //     uint32 J = J_N;
  //     uint64 Sc = 0ull;
  //     uint64 Mc = 0ull;
  //     for (uint32 i = 0; i < N; i++)
  //     {
  //       Sc |= (I & 1ull << i) << i | (J & 1ull << i) << (i + 1);
  //     }

  //     uint64 Mc0 = (1ull << (NMAX + NMAX - N - N)) - 1;
  //     uint64 detaMc = 0ull;
  //     detaMc = Sc << (NMAX + NMAX + 1 - N - N);
  //     Mc = Mc0 + detaMc;
  //     return Mc;
  //   }

  __host__ __device__ void m_Mc_toIJN(uint64 Mc, uint32 &I, uint32 &J, uint32 &N)
  {
    N = m_NofMc(Mc);
    uint64 Mc0 = (1ull << (NMAX + NMAX - N - N)) - 1;
    uint64 deta = 1ull << (NMAX + NMAX + 1 - N - N);
    uint64 Sc = (Mc - Mc0) >> (NMAX + NMAX + 1 - N - N);

    I = 0ul;
    J = 0ul;
    for (int temp_i = 0; temp_i < 2 * NMAX; temp_i++)
    {
      I |= ((Sc & (1ull << temp_i)) >> (temp_i >> 1));
      temp_i++;
      J |= ((Sc & (1ull << temp_i)) >> ((temp_i >> 1) + 1));
    }
  }

  // void m_Mc_toIJN(uint64 Mc, uint32 &I, uint32 &J, uint32 &N)
  //   {
  //     N = m_NofMc(Mc);
  //     uint64 Mc0 = (1ull << (NMAX + NMAX - N - N)) - 1;
  //     uint64 deta = 1ull << (NMAX + NMAX + 1 - N - N);
  //     uint64 Sc = (Mc - Mc0) >> (NMAX + NMAX + 1 - N - N);

  //     I = 0ul;
  //     J = 0ul;
  //     for (int temp_i = 0; temp_i < 2 * NMAX; temp_i++)
  //     {
  //       I |= ((Sc & (1ull << temp_i)) >> (temp_i >> 1));
  //       temp_i++;
  //       J |= ((Sc & (1ull << temp_i)) >> ((temp_i >> 1) + 1));
  //     }
  // }

  __device__ unsigned long long int m_FMcone(uint64 Mc)
  {
    unsigned long long int FMc;
    uint32 N = 0u;
    N = m_NofMc(Mc);
    uint32 NF = N - 1;
    if (NF > N || N == 0)
    {
      FMc = Mc;
      return FMc;
    }
    else
    {
      return m_FMcNF(Mc, NF);
    }
  }

  __host__ __device__ unsigned long long int m_FMcNF(uint64 Mc, uint32 NF)
  {
    unsigned long long int FMc;
    uint32 N = 0u;
    N = m_NofMc(Mc);

    if (NF > N || NF < 0)
    {
      FMc = Mc;
      return FMc;
    }
    if (N == NF)
    {
      FMc = Mc;
    }
    else
    {
      uint64 FMc0 = 0ull;
      FMc0 = (1ull << (NMAX + NMAX - NF - NF)) - 1;
      uint64 detaFMc = 0ull;
      detaFMc = (Mc >> (NMAX + NMAX + 1 - NF - NF))
                << (NMAX + NMAX + 1 - NF - NF);
      FMc = FMc0 + detaFMc;
    }
    return FMc;
  }

  __host__ __device__ void m_SonInterval(uint64 Mc, uint32 NS, uint64 &minson, uint64 &maxson)
  {
    uint32 N = m_NofMc(Mc);
    if (N >= NMAX || NS <= N)
    {
      minson = maxson = Mc;
      return;
    }
    uint64 Mc0 = 0ull;
    Mc0 = (1ull << (NMAX + NMAX - N - N)) - (1ull << (NMAX + NMAX - NS - NS));
    minson = Mc - Mc0;
    maxson = Mc + Mc0;
  }

} // namespace BASE_XYCode

#ifdef BASE_SpaceCode
using namespace BASE_XYCode;

__device__ LBGridCoor toLBGridCoor_from_LBPoint(LBPoint geopoint, int level)
{
  LBGridCoor gc;
  switch (d_ESPGCode)
  {
  case 4326: // tms model
    gc = toLBGridCoor_from_LBPoint_ESPG4326(geopoint, level);
    break;
  case 43260: // xyz model
    gc = toLBGridCoor_from_LBPoint_ESPG4326(geopoint, level);
    break;
  case 1:
    // cout << "toLBGridCoor_from_LBPoint_ESPG1 undone" << endl;
    break;
  default:
    break;
  }
  return gc;
}

__device__ uint64 toLBCode_from_LBPoint(LBPoint geopoint, int level)
{
  LBGridCoor gc = toLBGridCoor_from_LBPoint(geopoint, level);
  return m_IJN_toMc(gc.coI, gc.coJ, gc.level);
}

//  uint64 toLBCode_from_LBGridCoor(LBGridCoor coor)
// {
//   return m_IJN_toMc(coor.coI, coor.coJ, coor.level);
// }

__host__ __device__ uint64 toLBCode_from_LBGridCoor(LBGridCoor coor)
{
  return m_IJN_toMc(coor.coI, coor.coJ, coor.level);
}

__host__ __device__ LBGridCoor toLBGridCoor_from_LBCode(uint64 code)
{
  LBGridCoor coor;
  m_Mc_toIJN(code, coor.coI, coor.coJ, coor.level);
  return coor;
}

//   LBGridCoor toLBGridCoor_from_LBCode(uint64 code)
// {
//   LBGridCoor coor;
//   m_Mc_toIJN(code, coor.coI, coor.coJ, coor.level);
//   return coor;
// }

__host__ __device__ LBRange toLBRange_from_LBCode(uint64 code)
{
#ifdef __CUDA__ARCH__
  LBGridCoor coor = toLBGridCoor_from_LBCode(code);
  LBRange range;
  switch (*d_ESPGCode)
  {
  case 4326: // tms model
    range = toLBRange_from_LBGridCoor_ESPG4326(coor);
    break;
  case 43260: // xyz model
    range = toLBRange_from_LBGridCoor_ESPG4326(coor);
    break;
  case 1:
    // cout << "toLBGridCoor_from_LBPoint_ESPG1 undone" << endl;
    break;
  default:
    break;
  }
  return range;
#else
  LBGridCoor coor = toLBGridCoor_from_LBCode(code);
  LBRange range;
  switch (d_ESPGCode)
  {
  case 4326: // tms model
    range = toLBRange_from_LBGridCoor_ESPG4326(coor);
    break;
  case 43260: // xyz model
    range = toLBRange_from_LBGridCoor_ESPG4326(coor);
    break;
  case 1:
    // cout << "toLBGridCoor_from_LBPoint_ESPG1 undone" << endl;
    break;
  default:
    break;
  }
  return range;
#endif
}

// vector<LBPoint> toLBRange4Points_from_LBCode(uint64 code)
// {
//   LBRange range = toLBRange_from_LBCode(code);
//   vector<LBPoint> points;
//   points.resize(4);
//   points[0] = {range.lon_min, range.lat_min};
//   points[1] = {range.lon_min, range.lat_max};
//   points[2] = {range.lon_max, range.lat_max};
//   points[3] = {range.lon_max, range.lat_min};
//   return points;
// }

__device__ LBPoint toLBPoint_from_CenterOfLBCode(uint64 code)
{
  LBRange range;
  range = toLBRange_from_LBCode(code);
  LBPoint geopoint;
  geopoint.lon = 0.5 * (range.lon_min + range.lon_max);
  geopoint.lat = 0.5 * (range.lat_min + range.lat_max);
  return geopoint;
}

__device__ uint32 getLevel_ofLBCode(uint64 code) { return m_NofMc(code); }

__device__ uint64 getParent_ofLBCode(uint64 code, uint32 plevel)
{
  return m_FMcNF(code, plevel);
}
__device__ uint64 getParent_ofLBCode(uint64 code) { return m_FMcone(code); }

//@return - 1[non - adjacent] 0[adjacent] 1[code1 contains code2] 2[code2
// contains code1] 3[code1 == code2]
__device__ int m_isAdjacentOfTwoCodes(uint64 code1, uint64 code2)
{
  LBGridCoor gc1 = toLBGridCoor_from_LBCode(code1);
  LBGridCoor gc2 = toLBGridCoor_from_LBCode(code2);
  int detaI = abs((int)gc1.coI - (int)gc2.coI);
  int detaJ = abs((int)gc1.coJ - (int)gc2.coJ);
  if (detaI == 1 || detaJ == 1)
  {
    return 0;
  }
  else
  {
    return -1;
  }
}

__device__ int getRealationshipOfTwoCodes(uint64 code1, uint64 code2)
{
  if (code1 == code2)
  {
    return 3;
  }
  uint32 level1 = getLevel_ofLBCode(code1);
  uint32 level2 = getLevel_ofLBCode(code2);
  if (level1 == level2)
  {
    return m_isAdjacentOfTwoCodes(code1, code2);
  }
  else
  {
    if (level1 < level2)
    {
      code2 = getParent_ofLBCode(code2, level1);
      if (code1 == code2)
      {
        return 1;
      }
      else
      {
        return m_isAdjacentOfTwoCodes(code1, code2);
      }
    }
    else
    {
      code1 = getParent_ofLBCode(code1, level2);
      if (code1 == code2)
      {
        return 2;
      }
      else
      {
        return m_isAdjacentOfTwoCodes(code1, code2);
      }
    }
  }
}

__host__ __device__ uint64 m_Panduan_FMc_Consistedof4SMc(uint64 A, uint64 B, uint64 C,
                                                uint64 D)
{
  uint32 N = m_NofMc(A);
  uint64 deta = 1ull << (NMAX + NMAX + 1 - N - N);
  if ((B - A == deta) && (C - B == deta) && (D - C == deta))
  {

    uint64 FMc;
    if (N >= 1)
    {
      FMc = m_FMcNF(A, N - 1);
    }
    else
    {
      FMc = m_FMcNF(A, 0);
    }

    return FMc;
  }
  else
  {

    return 1ull;
  }
}

// vector<uint64> toMultiscaleCodes_fromSinglescaleCodes(vector<uint64> codes)
// {
//   vector<uint64> vMc0;
//   vector<uint64> vMc1;
//   vector<uint64> vMcM;

//   vMc0.insert(vMc0.end(), codes.begin(), codes.end());

//   sort(vMc0.begin(), vMc0.end());

//   int k = 0;
//   uint64 Mc_0 = 0;
//   uint64 FMc_0 = 0;
//   uint32 N_0 = 0;
//   uint64 Mc_0_A = 0;
//   uint64 Mc_0_B = 0;

//   for (int i = 0; (int)vMc0.size() != 0; i++)
//   {
//     for (k = 0; k < (int)vMc0.size();)
//     {

//       Mc_0 = vMc0[k];
//       N_0 = m_NofMc(Mc_0);
//       if (N_0 >= 1)
//       {
//         FMc_0 = m_FMcNF(Mc_0, N_0 - 1);
//       }
//       else
//       {
//         FMc_0 = m_FMcNF(Mc_0, 0);
//       }

//       m_SonInterval(FMc_0, N_0, Mc_0_A, Mc_0_B);

//       if (Mc_0 == Mc_0_A && ((int)vMc0.size() - k - 1) >= 3)
//       {

//         FMc_0 = m_Panduan_FMc_Consistedof4SMc(vMc0[k], vMc0[k + 1], vMc0[k + 2],
//                                               vMc0[k + 3]);
//         if (FMc_0 == 1)
//         {
//           vMcM.push_back(vMc0[k]);
//           k = k + 1;
//         }
//         else
//         {

//           vMc1.push_back(FMc_0);
//           k = k + 4;
//         }
//       }
//       else
//       {
//         vMcM.push_back(vMc0[k]);
//         k = k + 1;
//       }
//     }

//     vMc0.clear();
//     if ((int)vMc1.size() > 0)
//     {
//       vMc0.insert(vMc0.end(), vMc1.begin(), vMc1.end());
//       vMc0.insert(vMc0.end(), vMcM.begin(), vMcM.end());
//       sort(vMc0.begin(), vMc0.end());
//       vMc1.clear();
//       vMcM.clear();
//     }
//     else
//     {
//       break;
//     }
//   } // i
//   vMc0.clear();
//   vMc1.clear();
//   return vMcM;
// }

vector<uint64> m_Sort_DeleteSameCodes(vector<uint64> vcodes)
{
  for (auto & item : vcodes) {
    printf("%llu ****", item);
  }

  set<uint64> codeset(vcodes.begin(), vcodes.end());
  vector<uint64> vnew;
  vnew.assign(codeset.begin(), codeset.end());
  return vnew;
}

#endif // BASE_SpaceCode

#define ADVANCE_SpaceCod
#ifdef ADVANCE_SpaceCode
/*
 * host function void getCodesOfOneLine_Bresenham(LBPoint *points, uint32 level,
                                                    )
*/
struct coorXY
{
  double X = 0.0;
  double Y = 0.0;
};

// static double rayBoxIntersection_2D(coorXY origin, coorXY direction,
//                                     coorXY vmin, coorXY vmax)
// {
//   double tmin, tmax, tymin, tymax;

//   if (direction.X >= 0)
//   {
//     tmin = (vmin.X - origin.X) / direction.X;
//     tmax = (vmax.X - origin.X) / direction.X;
//   }
//   else
//   {
//     tmin = (vmax.X - origin.X) / direction.X;
//     tmax = (vmin.X - origin.X) / direction.X;
//   }

//   if (direction.Y >= 0)
//   {
//     tymin = (vmin.Y - origin.Y) / direction.Y;
//     tymax = (vmax.Y - origin.Y) / direction.Y;
//   }
//   else
//   {
//     tymin = (vmax.Y - origin.Y) / direction.Y;
//     tymax = (vmin.Y - origin.Y) / direction.Y;
//   }

//   if ((tmin > tymax) || (tymin > tmax))
//   {

//     tmin = -1;
//     return tmin;
//   }

//   if (tymin > tmin)
//   {
//     tmin = tymin;
//   }

//   if (tymax < tmax)
//   {
//     tmax = tymax;
//   }

//   return tmin;
// }
// static void amanatidesWooAlgorithm_2D(coorXY origin, coorXY end,
//                                       coorXY minBound, coorXY maxBound, int nx,
//                                       int ny, vector<LBGridCoor> &ij_list)
// {
//   ij_list.clear();
//   coorXY direction;
//   direction.X = end.X - origin.X;
//   direction.Y = end.Y - origin.Y;

//   double tmin = rayBoxIntersection_2D(origin, direction, minBound, maxBound);
//   if (tmin < 0)
//   {
//     tmin = 0;
//   }
//   coorXY start;
//   start.X = origin.X + tmin * direction.X;
//   start.Y = origin.Y + tmin * direction.Y;

//   coorXY boxSize;
//   boxSize.X = maxBound.X - minBound.X;
//   boxSize.Y = maxBound.Y - minBound.Y;

//   int x = int(floor(((start.X - minBound.X) / boxSize.X) * nx) + 1);
//   int y = int(floor(((start.Y - minBound.Y) / boxSize.Y) * ny) + 1);

//   if (x == (nx + 1))
//     x = x - 1;
//   if (y == (ny + 1))
//     y = y - 1;

//   double tVoxelX, tVoxelY;
//   int stepX, stepY;
//   if (direction.X >= 0)
//   {
//     tVoxelX = double(x) / double(nx);
//     stepX = 1;
//   }
//   else
//   {
//     tVoxelX = double(x - 1) / double(nx);
//     stepX = -1;
//   }

//   if (direction.Y >= 0)
//   {
//     tVoxelY = double(y) / double(ny);
//     stepY = 1;
//   }
//   else
//   {
//     tVoxelY = double(y - 1) / double(ny);
//     stepY = -1;
//   }

//   double voxelMaxX, voxelMaxY;
//   voxelMaxX = minBound.X + tVoxelX * boxSize.X;
//   voxelMaxY = minBound.Y + tVoxelY * boxSize.Y;
//   double tMaxX, tMaxY;
//   tMaxX = tmin + (voxelMaxX - start.X) / direction.X;
//   tMaxY = tmin + (voxelMaxY - start.Y) / direction.Y;

//   double voxelSizeX = boxSize.X / nx;
//   double voxelSizeY = boxSize.Y / ny;
//   double tDeltaX = voxelSizeX / abs(direction.X);
//   double tDeltaY = voxelSizeY / abs(direction.Y);

//   while ((x <= nx) && (x >= 1) && (y <= ny) && (y >= 1))
//   {
//     LBGridCoor tt;
//     tt.coI = (uint32)x;
//     tt.coJ = (uint32)y;
//     ij_list.push_back(tt);

//     if (tMaxX < tMaxY)
//     {
//       x = x + stepX;
//       tMaxX = tMaxX + tDeltaX;
//     }
//     else
//     {
//       y = y + stepY;
//       tMaxY = tMaxY + tDeltaY;
//     }
//   }
// }

// //@brief:  Improved DDA (no leakage)
// static void getCodesOfOneLine_ImprovedDDA(LBPoint point1, LBPoint point2,
//                                           uint32 level, uint64 &code1,
//                                           vector<uint64> &allcodes,
//                                           uint64 &code2)
// {
//   allcodes.clear();
//   code1 = toLBCode_from_LBPoint(point1, level);
//   code2 = toLBCode_from_LBPoint(point2, level);
//   LBGridCoor coor1 = toLBGridCoor_from_LBPoint(point1, MAXLEVEL_LB);
//   LBGridCoor coor2 = toLBGridCoor_from_LBPoint(point2, MAXLEVEL_LB);

//   coorXY origin, end;
//   double deta = double(pow(2, MAXLEVEL_LB - level));
//   origin = {(double)coor1.coI / deta, (double)coor1.coJ / deta};
//   end = {(double)coor2.coI / deta, (double)coor2.coJ / deta};

//   coorXY minBound, maxBound;
//   minBound.X = floor(min(origin.X, end.X));
//   minBound.Y = floor(min(origin.Y, end.Y));
//   maxBound.X = ceil(max(origin.X, end.X));
//   maxBound.Y = ceil(max(origin.Y, end.Y));

//   if (maxBound.X == minBound.X)
//   {
//     maxBound.X = maxBound.X + 1;
//   }
//   if (maxBound.Y == minBound.Y)
//   {
//     maxBound.Y = maxBound.Y + 1;
//   }

//   vector<LBGridCoor> ij_list;
//   int nx, ny;
//   nx = int(maxBound.X - minBound.X);
//   ny = int(maxBound.Y - minBound.Y);

//   amanatidesWooAlgorithm_2D(origin, end, minBound, maxBound, nx, ny, ij_list);

//   LBGridCoor realcoor;

//   for (int i = 0; i < ij_list.size(); i++)
//   {
//     realcoor.coI = ij_list[i].coI + uint32(floor(minBound.X)) - 1;
//     realcoor.coJ = ij_list[i].coJ + uint32(floor(minBound.Y)) - 1;
//     realcoor.level = level;
//     allcodes.push_back(toLBCode_from_LBGridCoor(realcoor));
//   }
// }

// vector<uint64> getCodesOfLines_fixedlevel(vector<LBPoint> coorlist,
//                                           uint32 level, bool isBresenham) {
//   vector<uint64> vcodes;
//   if ((int)coorlist.size() < 2) {
//     printf("error: Point num < 2 \n");
//     vcodes.clear();
//     return vcodes;
//   }

//   level = min(MAXLEVEL_LB, max(0, level));

//   if (isBresenham) {
//     uint64 code1 = 0ull;
//     uint64 code2 = 0ull;
//     for (int i = 0; i < (int)coorlist.size() - 1; i++) {
//       vector<uint64> midcodes;
//       getCodesOfOneLine_Bresenham(coorlist[i], coorlist[i + 1], level, code1,
//                                   midcodes, code2);
//       vcodes.push_back(code1);
//       vcodes.insert(vcodes.end(), midcodes.begin(), midcodes.end());
//     }
//     vcodes.push_back(code2);
//   } else { // Improved DDA
//     for (int i = 0; i < (int)coorlist.size() - 1; i++) {
//       vector<uint64> allcodes;
//       uint64 code1 = 0ull;
//       uint64 code2 = 0ull;
//       getCodesOfOneLine_ImprovedDDA(coorlist[i], coorlist[i + 1], level, code1,
//                                     allcodes, code2);
//       vcodes.insert(vcodes.end(), allcodes.begin(), allcodes.end());
//       vcodes.push_back(code2);
//     }
//   }

//   bool isSortDeleteSame = true;
//   if (isSortDeleteSame) {
//     vcodes = m_Sort_DeleteSameCodes(vcodes);
//   }

//   return vcodes;
// }

__device__ double rayBoxIntersection_2D(coorXY origin, coorXY direction,
                                        coorXY vmin, coorXY vmax)
{
  double tmin, tmax, tymin, tymax;

  if (direction.X >= 0)
  {
    tmin = (vmin.X - origin.X) / direction.X;
    tmax = (vmax.X - origin.X) / direction.X;
  }
  else
  {
    tmin = (vmax.X - origin.X) / direction.X;
    tmax = (vmin.X - origin.X) / direction.X;
  }

  if (direction.Y >= 0)
  {
    tymin = (vmin.Y - origin.Y) / direction.Y;
    tymax = (vmax.Y - origin.Y) / direction.Y;
  }
  else
  {
    tymin = (vmax.Y - origin.Y) / direction.Y;
    tymax = (vmin.Y - origin.Y) / direction.Y;
  }

  if ((tmin > tymax) || (tymin > tmax))
  {

    tmin = -1;
    return tmin;
  }

  if (tymin > tmin)
  {
    tmin = tymin;
  }

  if (tymax < tmax)
  {
    tmax = tymax;
  }

  return tmin;
}

__device__ void addToIJist(LBGridCoor *ij_list, int &count, LBGridCoor newCode)
{
  if (count < MAX_MIDCODES)
  {
    ij_list[count] = newCode;
    count++;
  }
}

__device__ void clearIJList(LBGridCoor *ij_list, uint64 &count)
{
  count = 0;
}

__device__ void amanatidesWooAlgorithm_2D(coorXY origin, coorXY end,
                                          coorXY minBound, coorXY maxBound, int nx,
                                          int ny, LBGridCoor *ij_list,
                                          uint64 &ij_list_Num)
{
  // ij_list.clear();
  clearIJList(ij_list, ij_list_Num);
  coorXY direction;
  direction.X = end.X - origin.X;
  direction.Y = end.Y - origin.Y;

  double tmin = rayBoxIntersection_2D(origin, direction, minBound, maxBound);
  if (tmin < 0)
  {
    tmin = 0;
  }
  coorXY start;
  start.X = origin.X + tmin * direction.X;
  start.Y = origin.Y + tmin * direction.Y;

  coorXY boxSize;
  boxSize.X = maxBound.X - minBound.X;
  boxSize.Y = maxBound.Y - minBound.Y;

  int x = int(floor(((start.X - minBound.X) / boxSize.X) * nx) + 1);
  int y = int(floor(((start.Y - minBound.Y) / boxSize.Y) * ny) + 1);

  if (x == (nx + 1))
    x = x - 1;
  if (y == (ny + 1))
    y = y - 1;

  double tVoxelX, tVoxelY;
  int stepX, stepY;
  if (direction.X >= 0)
  {
    tVoxelX = double(x) / double(nx);
    stepX = 1;
  }
  else
  {
    tVoxelX = double(x - 1) / double(nx);
    stepX = -1;
  }

  if (direction.Y >= 0)
  {
    tVoxelY = double(y) / double(ny);
    stepY = 1;
  }
  else
  {
    tVoxelY = double(y - 1) / double(ny);
    stepY = -1;
  }

  double voxelMaxX, voxelMaxY;
  voxelMaxX = minBound.X + tVoxelX * boxSize.X;
  voxelMaxY = minBound.Y + tVoxelY * boxSize.Y;
  double tMaxX, tMaxY;
  tMaxX = tmin + (voxelMaxX - start.X) / direction.X;
  tMaxY = tmin + (voxelMaxY - start.Y) / direction.Y;

  double voxelSizeX = boxSize.X / nx;
  double voxelSizeY = boxSize.Y / ny;
  double tDeltaX = voxelSizeX / abs(direction.X);
  double tDeltaY = voxelSizeY / abs(direction.Y);

  while ((x <= nx) && (x >= 1) && (y <= ny) && (y >= 1))
  {
    LBGridCoor tt;
    tt.coI = (uint32)x;
    tt.coJ = (uint32)y;
    ij_list[ij_list_Num] = (tt);

    if (tMaxX < tMaxY)
    {
      x = x + stepX;
      tMaxX = tMaxX + tDeltaX;
    }
    else
    {
      y = y + stepY;
      tMaxY = tMaxY + tDeltaY;
    }

    ij_list_Num++;
  }
}

__device__ void addToMidcodes(uint64 *midcodes, int &count, uint64 newCode)
{
  if (count < MAX_MIDCODES)
  {
    midcodes[count] = newCode;
    count++;
  }
}

__device__ void clearMidcodes(uint64 *midcodes, uint64 &count)
{
  count = 0;
}

__device__ void addToAllcodes(uint64 *allcodes, int &count, uint64 newCode)
{
  if (count < MAX_ALLCODES)
  {
    allcodes[count] = newCode;
    count++;
  }
}

__device__ void clearAllcodes(uint64 *allcodes, uint64 &count)
{
  count = 0;
}

__device__ void getCodesOfOneLine_Bresenham(LBPoint point1, LBPoint point2,
                                            uint32 level, uint64 &code1,
                                            uint64 *midcodes,
                                            uint64 &midcodesCount,
                                            uint64 &code2)
{

  // midcodes.clear();
  clearMidcodes(midcodes, midcodesCount);
  code1 = toLBCode_from_LBPoint(point1, level);
  code2 = toLBCode_from_LBPoint(point2, level);
  LBGridCoor coor1 = toLBGridCoor_from_LBCode(code1);
  LBGridCoor coor2 = toLBGridCoor_from_LBCode(code2);

  int Ibegin = (int)coor1.coI;
  int Jbegin = (int)coor1.coJ;
  int Iend = (int)coor2.coI;
  int Jend = (int)coor2.coJ;
  int detaI = Iend - Ibegin;
  int detaJ = Jend - Jbegin;
  int absdi = abs(detaI);
  int absdj = abs(detaJ);
  int sumNFR = ((absdi >= absdj) ? absdi : absdj) - 1;

  if (sumNFR <= 0)
  {
    return;
  }

  uint32 *I = new uint32[sumNFR + 1];
  uint32 *J = new uint32[sumNFR + 1];
  *(I + 0) = Ibegin;
  *(J + 0) = Jbegin;
  for (int tempi = 1; tempi < sumNFR; tempi++)
  {
    *(I + tempi) = 0ul;
    *(J + tempi) = 0ul;
  }

  int fi = (detaI >= 0) ? 1 : -1;
  int fj = (detaJ >= 0) ? 1 : -1;
  int deta0 = 1;
  if (absdi >= absdj)
  {
    int p = 2 * absdj - absdi;
    for (int n = 1; n <= sumNFR; n++)
    {
      if (p < 0)
      {
        *(I + n) = *(I + n - 1) + fi * deta0;
        *(J + n) = *(J + n - 1);
        p = p + 2 * absdj;
      }
      else
      {
        *(I + n) = *(I + n - 1) + fi * deta0;
        *(J + n) = *(J + n - 1) + fj * deta0;
        p = p + 2 * absdj - 2 * absdi;
      }
    }
  }
  else
  {
    int p = 2 * absdi - absdj;
    for (int n = 1; n <= sumNFR; n++)
    {
      if (p < 0)
      {
        *(I + n) = *(I + n - 1);
        *(J + n) = *(J + n - 1) + fj * deta0;
        p = p + 2 * absdi;
      }
      else
      {
        *(I + n) = *(I + n - 1) + fi * deta0;
        *(J + n) = *(J + n - 1) + fj * deta0;
        p = p + 2 * absdi - 2 * absdj;
      }
    }
  }

  // midcodes.clear();
  clearMidcodes(midcodes, midcodesCount);
  for (int n = 1; n <= sumNFR; n++)
  {
    LBGridCoor ttcoor = {uint32(*(I + n)), uint32(*(J + n)), level};
    uint64 ttcode = toLBCode_from_LBGridCoor(ttcoor);
    midcodes[n] = (ttcode);
  }

  delete[] I;
  delete[] J;
  return;
}

__global__ void toLBCode_from_LBPoint_kernel(LBPoint *geopoints, int level, int num, uint64 *codes)
{
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadId >= num)
    return;
  LBPoint geopoint = geopoints[threadId];
  codes[threadId] = toLBCode_from_LBPoint(geopoint, level);
  // printf("threadId = %d, res = %ld\n", threadId, res[threadId]);
}

__device__ void getCodesOfOneLine_ImprovedDDA(LBPoint point1, LBPoint point2,
                                              uint32 level, uint64 &code1,
                                              uint64 *allcodes,
                                              uint64 &allcodeNum,
                                              uint64 &code2,
                                              int max_allcodes_num
                                              // LBGridCoor *ij_list
)
{
  // printf("current point1: [%lf, %lf], point2: [%lf, %lf]\n", point1.lat,
  //   point1.lon,
  //   point2.lat,
  //   point2.lon);
  // allcodes.clear();
  clearAllcodes(allcodes, allcodeNum);
  code1 = toLBCode_from_LBPoint(point1, level);
  code2 = toLBCode_from_LBPoint(point2, level);
  LBGridCoor coor1 = toLBGridCoor_from_LBPoint(point1, MAXLEVEL_LB);
  LBGridCoor coor2 = toLBGridCoor_from_LBPoint(point2, MAXLEVEL_LB);

  coorXY origin, end;
  double deta = double(pow(2, MAXLEVEL_LB - level));
  origin = {(double)coor1.coI / deta, (double)coor1.coJ / deta};
  end = {(double)coor2.coI / deta, (double)coor2.coJ / deta};

  // printf("origin: X- %lf, Y- %lf\n", origin.X, origin.Y);

  coorXY minBound, maxBound;
  minBound.X = floor(min(origin.X, end.X));
  minBound.Y = floor(min(origin.Y, end.Y));
  maxBound.X = ceil(max(origin.X, end.X));
  maxBound.Y = ceil(max(origin.Y, end.Y));

  if (maxBound.X == minBound.X)
  {
    maxBound.X = maxBound.X + 1;
  }
  if (maxBound.Y == minBound.Y)
  {
    maxBound.Y = maxBound.Y + 1;
  }

  uint64 ij_list_size = 0;
  // LBGridCoor ij_list[100];
  LBGridCoor *ij_list = (LBGridCoor*) malloc(max_allcodes_num * sizeof(LBGridCoor));
  

  int nx, ny;
  nx = int(maxBound.X - minBound.X);
  ny = int(maxBound.Y - minBound.Y);

  amanatidesWooAlgorithm_2D(origin, end, minBound, maxBound, nx, ny, ij_list, ij_list_size);

  LBGridCoor realcoor;
  for (int i = 0; i < ij_list_size; i++)
  {
    realcoor.coI = ij_list[i].coI + uint32(floor(minBound.X)) - 1;
    realcoor.coJ = ij_list[i].coJ + uint32(floor(minBound.Y)) - 1;
    realcoor.level = level;
    // allcodes.push_back(toLBCode_from_LBGridCoor(realcoor));
    allcodes[allcodeNum] = toLBCode_from_LBGridCoor(realcoor);
    allcodeNum++;
  }
  free(ij_list);
}

__global__ void modifyArray(int *array, int size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size)
  {
    array[tid] *= 2; // 对数组中的每个元素乘以2
  }
}

__global__ void toLBGridCoor_from_LBCode_Kernel(LBGridCoor *result, const uint64 *coorlist, uint64 size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x; // blockIdx表示 block编号  blockDim.x表示block的线程数  threadIdx表示线程编号
  if (index  < size) {
    result[index] = toLBGridCoor_from_LBCode(coorlist[index]);
    printf("h_borderCoor[%d].coI: %d, \
            h_borderCoor[%d].coJ: %d,  \
            h_borderCoor[%d].level: %d \n",
                 index, result[index].coI,
                 index, result[index].coJ,
                 index, result[index].level);
  }
}

__global__ void getCodesOfLinesKernel(const LBPoint *coorlist,
                                      int numPoints, uint32 level,
                                      bool isBresenham,
                                      uint64 *vcodes,
                                      uint64 &vcodesNum,
                                      // uint64 *allcodes,
                                      // LBGridCoor *d_ij_list,
                                      uint64 *uninformOutput4GPU,
                                      int max_allcodes_num)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x; // 计算线程的全局索引

  // printf("coorlist[%d]: [%lf][%lf]\n", coorlist[index].lat, coorlist[index].lon);
  if (index < numPoints - 1)
  {
    uint64 code1 = 0ull;
    uint64 code2 = 0ull;

    // printf("process data idx: %d %d\n", index, index + 1);

    // printf("data: [%lf, %lf] and next data: [%lf, %lf]\n", coorlist[index].lat,
    //        coorlist[index].lon,
    //        coorlist[index + 1].lat,
    //        coorlist[index + 1].lon);

    //  LBPoint point1 = coorlist[index];
    //  LBPoint point2 = coorlist[index + 1];

    uint64 midcodesCount = 0; // 记录已添加的中间点编码数量

    uint64 allcodesCount = 0; // 记录已添加的中间点编码数量

    if (isBresenham)
    {
      // 调用Bresenham算法生成线段两端点之间的编码及中间点编码
      //(point1, point2, level, code1, midcodes, midcodesCount, code2);
      //  vcodes[index] = code1;
      //  for (int i = 0; i < midcodesCount; i++)
      //  {
      //    //vcodes[index + 1 + i] = midcodes[i];
      //  }
    }
    else
    {
      LBPoint point1 = coorlist[index];
      LBPoint point2 = coorlist[index + 1];

      midcodesCount = 0;
      allcodesCount = 0;
      // 调用改进的DDA算法生成线段两端点之间的编码及中间点编码
      uint64 *allcodes = (uint64* )malloc(sizeof(uint64) * max_allcodes_num);

      getCodesOfOneLine_ImprovedDDA(point1, point2, level, code1, allcodes, allcodesCount, code2,max_allcodes_num); // 每个线程计算当前两个点之间的allcodesCount可能不同，所以考虑分开存储

      // uniformOutput4GPU
      uninformOutput4GPU[index * max_allcodes_num] = allcodesCount;

      // 设备端代码 数组之间的拷贝问题
      for (uint64 idx = 0; idx < allcodesCount; idx++)
      {
        // printf("Idx: %llu %llu ", idx, allcodes[idx]);
        if (index * max_allcodes_num + idx + 1 < numPoints * max_allcodes_num + max_allcodes_num) {
          uninformOutput4GPU[index * max_allcodes_num + idx + 1] = allcodes[idx];
        }
      }

      free(allcodes);
    }
  }
}

__global__ void test_kernel()
{
  int tid = threadIdx.x;
  int *tmp = (int *)malloc(10 * sizeof(int));
  for (int i = 0; i < 10; ++i)
  {
    tmp[i] = i + 1;
  }
  for (int i = 0; i < 10; ++i)
  {
    printf("tid[%d][%d] = %d\n", tid, i, tmp[i]);
  }

  free(tmp);
}

// __global__ void minMaxKernel(LBGridCoor *array, int *minI, int *maxI, int *minJ, int *maxJ, int size) {
//   // 线程索引
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;

//   if (tid < size) {
//     int coI = array[tid].coI;
//     int coJ = array[tid].coJ;

//     atomicMax(maxI, coI);
//     atomicMax(minI, coI);
//     atomicMin(maxJ, coJ);
//     atomicMin(minJ, coJ);
//   }
//   __syncthreads();
// }

// CUDA 核函数，用于计算数组中 coI 和 coJ 的最大值和最小值
__global__ void minMaxKernel(LBGridCoor *array, int *minI, int *maxI, int *minJ, int *maxJ, int size) {
    // 获取线程索引
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 初始化共享内存中的最小值和最大值
    __shared__ int sharedMinI;
    __shared__ int sharedMaxI;
    __shared__ int sharedMinJ;
    __shared__ int sharedMaxJ;

    // 第一个线程初始化共享内存中的值
    if (threadIdx.x == 0) {
        sharedMinI = array[0].coI;
        sharedMaxI = array[0].coI;
        sharedMinJ = array[0].coJ;
        sharedMaxJ = array[0].coJ;
    }
    __syncthreads();

    // 并行计算最小值和最大值
    while (tid < size) {
        atomicMin(&sharedMinI, array[tid].coI);
        atomicMax(&sharedMaxI, array[tid].coI);
        atomicMin(&sharedMinJ, array[tid].coJ);
        atomicMax(&sharedMaxJ, array[tid].coJ);
        tid += blockDim.x * gridDim.x;
    }
    __syncthreads();

    // 将结果写回全局内存
    if (threadIdx.x == 0) {
        *minI = sharedMinI;
        *maxI = sharedMaxI;
        *minJ = sharedMinJ;
        *maxJ = sharedMaxJ;
    }
}

__global__ void initSceneDataKernel(LBGridCoor *sceneData, int* wid_I, int *wid_J, int *Imin, int *Jmin) {
  // 线程索引
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < (*wid_I) * (*wid_J)) {
    int x = tid % (*wid_I);
    int y = tid / (*wid_I);
    sceneData[tid].coI = static_cast<uint32_t>(x + *Imin - 2);
    sceneData[tid].coJ = static_cast<uint32_t>(y + *Jmin - 2);
    sceneData[tid].level = 0;

    if (sceneData[tid].coI == 54 && sceneData[tid].coJ == 1085) {
      int a = 100;
    }

    if (x == 0 || x == *wid_I - 1 || y == 0 || y == *wid_J -1) {
      sceneData[tid].level = 1;
    }
  }
}


// __global__ void updateSceneDataKernel(LBGridCoor *sceneData, LBGridCoor * boder, int *wid_I, int * Jmin, int * Imin, int size) {
//   // 计算线程ID
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;

//   if (tid < size) {
//     int index = ((int)boder[tid].coI - *Jmin + 2) * (*wid_I) + (int)boder[tid].coI - *Imin + 2;
//     sceneData[index].level = 1;
//   }
// }

__global__ void setBorderLevelsKernel(LBGridCoor* sceneData, LBGridCoor* boder, int borderSize, int *wid_I, int *Imin, int *Jmin) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < borderSize) {
        printf("setBorderLevelsKernel-sceneData: %d %d %d\n", sceneData[tid].coI, sceneData[tid].coJ, sceneData[tid].level);
        int idx = (int)boder[tid].coI - *Imin + 2;
        int idy = (int)boder[tid].coJ - *Jmin + 2;
        int index = idy * (*wid_I) + idx;
        printf("---- %d %d %d\n", sceneData[index].coI, sceneData[index].coJ, sceneData[index].level);

       // printf("Thread value: %d %d %d %d\n", (int)boder[tid].coI, *Imin ,(int)boder[tid].coJ, *Jmin);

        if (62 == index) {
            int index = 62;
             printf("********* %llu %llu %llu\n", boder[tid].coI, sceneData[index].coJ, sceneData[index].level);
        }
        sceneData[index].level = 1;
    }
}

__global__ void searchLineNewSeedKernel(LBGridCoor* voxeldata, int J, int wid, int xLeft, int xRight, int len)
{
  int tid = blockIdx.x *blockDim.x + threadIdx.x;
  int xt = tid + xLeft;
  // 线程的个数 为xRight -xLeft + 1个
  if (tid < xRight -xLeft + 1 && (J != -1 && J != len + 1)) {
    bool findNewSeed = false;
    while (voxeldata[J *wid + xt].level == 0)
    {
      findNewSeed = true;
      xt++;
    }
    if (findNewSeed) {
      voxeldata[J * wid + xt - 1].level = 1;
    }
    while(xt <= xRight && voxeldata[J * wid + xt].level != 0) {
      xt++;
    }
  }
}


static int FillLineRight2(int I, int J, vector<LBGridCoor> &Voxeldata, int wid,
                          int len)
{
  int count = 1;
  while ((Voxeldata[J * wid + I + count].level != 1))
  {
    Voxeldata[J * wid + I + count].level = 1;
    count++;
  }
  return count;
}
static int FillLineleft2(int I, int J, vector<LBGridCoor> &Voxeldata, int wid,
                         int len)
{
  int count = 0;
  while ((Voxeldata[J * wid + I - count].level != 1))
  {
    Voxeldata[J * wid + I - count].level = 1;
    count++;
  }
  return count;
}
static void SearchLineNewSeed2(vector<LBGridCoor> &stk,
                               vector<LBGridCoor> &Voxeldata, int xLeft,
                               int xRight, int J, int wid, int len)
{
  if (J == -1 || J == len + 1)
    return;
  int xt = xLeft;
  bool findNewSeed = false;
  while (xt <= xRight)
  {
    findNewSeed = false;
    while (Voxeldata[J * wid + xt].level == 0)
    {
      findNewSeed = true;
      xt++;
    }
    if (findNewSeed)
    {
      Voxeldata[J * wid + xt - 1].level = 1;
      stk.push_back(Voxeldata[J * wid + xt - 1]);
      findNewSeed = false;
    }
    while (Voxeldata[J * wid + xt].level != 0 && (xt <= xRight))
    {
      xt++;
    }
  }
}

// __global__ void processSceneData(const LBGridCoor *sceneData, int dataSize, LBGridCoor *inner, uint32_t level_LB) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int insertIdx = 0; // Variable to track insertion index

//     if (idx < dataSize) {
//         if (sceneData[idx].level == 0) {
//             // Loop to find the first available index in inner
//             while (inner[insertIdx].level != 0 && insertIdx < dataSize) {
//                 insertIdx++;
//             }
//             if (insertIdx < dataSize) { // Make sure we haven't reached the end of inner
//                 inner[insertIdx].coI = sceneData[idx].coI;
//                 inner[insertIdx].coJ = sceneData[idx].coJ;
//                 inner[insertIdx].level = level_LB;
//             }
//         }
//     }
// }
vector<uint64> toMultiscaleCodes_fromSinglescaleCodes(vector<uint64> codes)
{
    vector<uint64> vMc0;
    vector<uint64> vMc1;
    vector<uint64> vMcM;

    vMc0.insert(vMc0.end(), codes.begin(), codes.end());

    sort(vMc0.begin(), vMc0.end());

    int k = 0;
    uint64 Mc_0 = 0;
    uint64 FMc_0 = 0;
    uint32 N_0 = 0;
    uint64 Mc_0_A = 0;
    uint64 Mc_0_B = 0;

    for (int i = 0; (int)vMc0.size() != 0; i++)
    {
        for (k = 0; k < (int)vMc0.size();)
        {

            Mc_0 = vMc0[k];
            N_0 = m_NofMc(Mc_0);
            if (N_0 >= 1)
            {
                FMc_0 = m_FMcNF(Mc_0, N_0 - 1);
            }
            else
            {
                FMc_0 = m_FMcNF(Mc_0, 0);
            }

            m_SonInterval(FMc_0, N_0, Mc_0_A, Mc_0_B);

            if (Mc_0 == Mc_0_A && ((int)vMc0.size() - k - 1) >= 3)
            {

                FMc_0 = m_Panduan_FMc_Consistedof4SMc(vMc0[k], vMc0[k + 1], vMc0[k + 2], vMc0[k + 3]);
                if (FMc_0 == 1)
                {
                    vMcM.push_back(vMc0[k]);
                    k = k + 1;
                }
                else
                {

                    vMc1.push_back(FMc_0);
                    k = k + 4;
                }
            }
            else
            {
                vMcM.push_back(vMc0[k]);
                k = k + 1;
            }
        }

        vMc0.clear();
        if ((int)vMc1.size() > 0)
        {
            vMc0.insert(vMc0.end(), vMc1.begin(), vMc1.end());
            vMc0.insert(vMc0.end(), vMcM.begin(), vMcM.end());
            sort(vMc0.begin(), vMc0.end());
            vMc1.clear();
            vMcM.clear();
        }
        else
        {
            break;
        }
    } // i
    vMc0.clear();
    vMc1.clear();
    return vMcM;
}

__global__ void toLBCode_from_LBGridCoor_Kernel(uint64* result, LBGridCoor * innercoor, int size, uint32 level) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size) {
      innercoor[idx].level = level;
      result[idx] = toLBCode_from_LBGridCoor(innercoor[idx]);
  }
}


vector<uint64> getCodesOfLines_fixedlevel_CUDA(vector<LBPoint> coorlist, 
                                                uint32 level,  
                                                vector<uint64> &innerCodes, 
                                                bool isBresenham, 
                                                bool ismultiscale,
                                                int max_vcodes_num,
                                                int max_allcodes_num)
{
  // coorlist大小
  int size = coorlist.size();
  // Device code
  LBPoint *dev_coorlist;
  cudaMalloc((LBPoint **)&dev_coorlist, size * sizeof(LBPoint)); // 设备端对应释放资源
  cudaMemcpy(dev_coorlist, coorlist.data(), size * sizeof(LBPoint), cudaMemcpyHostToDevice);

  // 这段代码实际DDA算法没有使用
  uint64 *h_vcodes = (uint64 *)malloc(max_vcodes_num * sizeof(uint64)); // 主机端对应释放资源
  uint64 *dev_vcodes;
  cudaMalloc((uint64 **)&dev_vcodes, max_vcodes_num * sizeof(uint64)); // 设备端对应释放资源

  level = min(MAXLEVEL_LB, max(0, level));

  // Launch the CUDA kernel
  int threadsPerBlock = 512;
  int numBlocks = (coorlist.size() + threadsPerBlock - 1) / threadsPerBlock;

  //printf("numBlocks: %d, threadPerBlock: %d\n", numBlocks, threadsPerBlock);

  uint64 *h_uniformOutput4GPU;
  h_uniformOutput4GPU = (uint64 *)malloc((size + 1) * max_allcodes_num * sizeof(uint64)); // 主机端对应释放资源
  memset(h_uniformOutput4GPU, 0, (size + 1) * max_allcodes_num  * sizeof(uint64));

  uint64 *uniformOutput4GPU;
  cudaMalloc(&uniformOutput4GPU, (size + 1) * max_allcodes_num  * sizeof(uint64)); // 设备端对应释放资源
  cudaMemcpy(uniformOutput4GPU, h_uniformOutput4GPU, (size + 1) * max_allcodes_num  * sizeof(uint64), cudaMemcpyHostToDevice);

  LBGridCoor *h_borderCoor = (LBGridCoor*) malloc((size + 1) * max_allcodes_num * sizeof(LBGridCoor)); // 主机端对应释放资源

  LBGridCoor *d_borderCoor;
  cudaMalloc(&d_borderCoor, (size + 1) * max_allcodes_num  *sizeof(LBGridCoor)); // 设备端对应释放资源
  cudaMemcpy(d_borderCoor, h_borderCoor, (size + 1) * max_allcodes_num  * sizeof(LBGridCoor), cudaMemcpyHostToDevice);

  uint64 vcodesNum = 0;

  // 创建CUDA流
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // outputs: uniforOutput4GPU 归一化了数据格式，便于存取
  getCodesOfLinesKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(dev_coorlist,
                                                              size,
                                                              level,
                                                              isBresenham,
                                                              (dev_vcodes),
                                                              vcodesNum,
                                                              // d_allcodes,
                                                              // d_ij_list,
                                                              uniformOutput4GPU,
                                                              max_allcodes_num);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  cudaFree(dev_coorlist); // 设备端释放dev_coorlist资源
  cudaFree(dev_vcodes); // 设备端释放dev_vcodes资源

  // 将上一个kernel执行结果uniforOutput4GPU拷贝到host端h_uniformOutput4GPU进行处理，组织成新的 h_boderCodes
  cudaMemcpy(h_uniformOutput4GPU, uniformOutput4GPU, (size + 1) * max_allcodes_num * sizeof(uint64), cudaMemcpyDeviceToHost);

  cudaFree(uniformOutput4GPU); // 设备端释放uniformOutput4GPU资源

  // 循环主机端h_uniformOutput4GPU，组织
  for (int i = 0; i < (size + 1) * max_allcodes_num; i += max_allcodes_num)
  {
    int allcode_count = h_uniformOutput4GPU[i];
    if (allcode_count == 0)
      continue;
    for (int j = 0; j < allcode_count; j++)
    {
      h_vcodes[vcodesNum] = h_uniformOutput4GPU[i + 1 + j];
      //printf("%llu ", h_vcodes[vcodesNum]);
      vcodesNum++;
    }
    //printf("\n");
  }

  free(h_uniformOutput4GPU); // 主机端释放h_uniformOutput4GPU资源

  /******************************在主机端组织好vector<uint64>h_boderCodes并完成处理***********************************/
  std::vector<uint64> h_borderCodes;

   for (int i = 0 ; i < vcodesNum; i++) {
    h_borderCodes.push_back(h_vcodes[i]);
    //printf("%llu\n", h_borderCodes[i]);
  }

  free(h_vcodes);// 释放h_vcodes资源

  bool isSortDeleteSame = true;
  if (isSortDeleteSame)
  {
    h_borderCodes = m_Sort_DeleteSameCodes(h_borderCodes);
  }

  uint64 *d_borderCodes;
  cudaMalloc(&d_borderCodes, sizeof(uint64) * h_borderCodes.size()); // 设备端资源释放
  cudaMemcpy(d_borderCodes, h_borderCodes.data(), sizeof(uint64) * h_borderCodes.size(), cudaMemcpyHostToDevice);

  /**************************************************************************************************************/
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  numBlocks = (h_borderCodes.size() + threadsPerBlock - 1) / threadsPerBlock;
  toLBGridCoor_from_LBCode_Kernel<<<numBlocks, threadsPerBlock, 0, stream1>>>(d_borderCoor, d_borderCodes, h_borderCodes.size());

  cudaDeviceSynchronize();
  /***********************************ScanLineSeedFill_2D parallelization****************************************/

  int h_Imax, h_Imin;
  int h_Jmax, h_Jmin;
  int *d_Imax, *d_Imin;
  int *d_Jmax, *d_Jmin;

  // 在设备（GPU）上分配内存
  cudaMalloc((void**)&d_Imax, sizeof(int));
  cudaMalloc((void**)&d_Imin, sizeof(int));

  cudaMalloc((void**)&d_Jmax, sizeof(int));
  cudaMalloc((void**)&d_Jmin, sizeof(int));

  // 初始化最大最小值为数组中的第一个元素
  cudaMemcpy(d_Imax, &h_borderCoor[0].coI, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Imin, &h_borderCoor[0].coI, sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_Jmax, &h_borderCoor[0].coJ, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Jmin, &h_borderCoor[0].coJ, sizeof(int), cudaMemcpyHostToDevice);

  // 计算最大值和最小值
  numBlocks = (h_borderCodes.size() + threadsPerBlock - 1) / threadsPerBlock;
  minMaxKernel<<<numBlocks, threadsPerBlock, 0, stream1>>>(d_borderCoor, d_Imin, d_Imax, d_Jmin, d_Jmax, h_borderCodes.size());

  // 添加同步点，等待核函数执行完成
  cudaStreamSynchronize(stream1);

  cudaStreamDestroy(stream1);

  cudaMemcpy(h_borderCoor, d_borderCoor, (size + 1) * max_allcodes_num *sizeof(LBGridCoor), cudaMemcpyDeviceToHost);

  for (int i =0; i < max_allcodes_num; i++) {
    printf("333####%d %d %d\n", h_borderCoor[i].coI,h_borderCoor[i].coJ, h_borderCoor[i].level);
  }

  LBGridCoor *d_borderCoor1;
  cudaMalloc(&d_borderCoor1, (size + 1) * max_allcodes_num *sizeof(LBGridCoor));

  cudaError_t error = cudaMemcpy(d_borderCoor1, h_borderCoor, (size + 1) * max_allcodes_num * sizeof(LBGridCoor), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }

  cudaFree(d_borderCoor); //设备端释放d_borderCoor资源

  // 将设备（GPU）上的最大最小值复制到主机内存
  cudaMemcpy(&h_Imax, d_Imax, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_Imin, d_Imin, sizeof(int), cudaMemcpyDeviceToHost);

  cudaMemcpy(&h_Jmax, d_Jmax, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_Jmin, d_Jmin, sizeof(int), cudaMemcpyDeviceToHost);

  // 输出最大值和最小值
  printf("IMax: %llu\n", h_Imax);
  printf("IMin: %llu\n", h_Imin);

  printf("JMax: %llu\n", h_Jmax);
  printf("JMin: %llu\n", h_Jmin);

  int wid_I = h_Imax - h_Imin + 4 + 1;
  int wid_J = h_Jmax - h_Jmin + 4 + 1;

  int *d_wid_I;
  int *d_wid_J;

  cudaMalloc(&d_wid_I, sizeof(int));
  cudaMalloc(&d_wid_J, sizeof(int));

  cudaMemcpy(d_wid_I, &wid_I, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wid_J, &wid_J, sizeof(int), cudaMemcpyHostToDevice);

  vector<LBGridCoor> SceneData;
  SceneData.resize(wid_I * wid_J);

  LBGridCoor *d_SceneData;
  cudaMalloc(&d_SceneData, sizeof(LBGridCoor) * wid_I * wid_J);

  cudaStream_t stream2;  // 创建了一个CUDA流 stream2
  cudaStreamCreate(&stream2);
  // 计算initSceneDataKernel参数
  numBlocks = (SceneData.size() + threadsPerBlock - 1) / threadsPerBlock;
  initSceneDataKernel<<<numBlocks, threadsPerBlock, 0, stream2>>>(d_SceneData,
                                                                  d_wid_I,
                                                                  d_wid_J,
                                                                  d_Imin,
                                                                  d_Jmin);  // output d_SceneData

  cudaStreamSynchronize(stream2);

  // 计算setBorderLevelsKernel参数
  numBlocks = (h_borderCodes.size() + threadsPerBlock - 1) / threadsPerBlock;
  setBorderLevelsKernel<<<numBlocks, threadsPerBlock, 0, stream2>>>(d_SceneData,
                                                                    d_borderCoor1,
                                                                    h_borderCodes.size(),
                                                                    d_wid_I,
                                                                    d_Imin,
                                                                    d_Jmin); // output d_SceneData

  // 添加同步点，等待第二个核函数执行完成
  cudaStreamSynchronize(stream2);

  // 销毁CUDA流
  cudaStreamDestroy(stream2);

  cudaFree(d_borderCoor1);

  cudaMemcpy(&SceneData[0], d_SceneData, sizeof(LBGridCoor) * wid_I * wid_J, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < wid_I * wid_J; i++) {
  //   printf("********* %llu %llu %llu\n", SceneData[i].coI, SceneData[i].coJ, SceneData[i].level);
  // }

    vector<LBGridCoor> Stk;
		LBGridCoor StartPoint;
		StartPoint.coI = 1 + h_Imin - 2;
		StartPoint.coJ = 1 + h_Jmin - 2;
		StartPoint.level = 1;

		Stk.push_back(StartPoint);

		while (!Stk.empty()) {

			LBGridCoor SeedPoint = Stk[(int)Stk.size() - 1];
			Stk.pop_back();
			SceneData[((int)SeedPoint.coJ - h_Jmin + 2) * wid_I + (int)SeedPoint.coI - h_Imin + 2].level = 1;

			int count = FillLineRight2((int)SeedPoint.coI - h_Imin + 2, (int)SeedPoint.coJ - h_Jmin + 2, SceneData, wid_I, wid_J);
      int xRight = (int)SeedPoint.coI - h_Imin + 2 + count - 1;
			count = FillLineleft2((int)SeedPoint.coI - 1 - h_Imin + 2, (int)SeedPoint.coJ - h_Jmin + 2, SceneData, wid_I, wid_J);
			int xLeft = (int)SeedPoint.coI - count - h_Imin + 2;

			SearchLineNewSeed2(Stk, SceneData, xLeft, xRight, (int)SeedPoint.coJ - 1 - h_Jmin + 2, wid_I, wid_J);
			SearchLineNewSeed2(Stk, SceneData, xLeft, xRight, (int)SeedPoint.coJ + 1 - h_Jmin + 2, wid_I, wid_J);
		}

    std::vector<LBGridCoor> inner;

		uint32 level_LB = h_borderCoor[0].level;
		for (int i = 0; i < SceneData.size(); i++) {
			if (SceneData[i].level == 0) {
				LBGridCoor temp_coor;
				temp_coor.coI = SceneData[i].coI;
				temp_coor.coJ = SceneData[i].coJ;
				temp_coor.level = level_LB;
				inner.push_back(temp_coor);
			}
		}

  free(h_borderCoor); // 主机端释放h_borderCoor资源
  cudaFree(d_SceneData); // 设备端释放 d_SceneData资源

  // 清理内存
  cudaFree(d_Imax);
  cudaFree(d_Imin);
  cudaFree(d_Jmax);
  cudaFree(d_Jmin);

  /***********************************ScanLineSeedFill_2D parallelization****************************************/

  // vector<uint64> innerCodes;
  innerCodes.resize((int)inner.size());

  uint64* d_innerCodes;
  cudaMalloc((void**)&d_innerCodes, innerCodes.size() * sizeof(uint64));
  LBGridCoor *d_inner;
  cudaMalloc((void**)&d_inner, sizeof(LBGridCoor)* inner.size());
  cudaMemcpy(d_inner, inner.data(), inner.size() * sizeof(LBGridCoor), cudaMemcpyHostToDevice);

  numBlocks = (innerCodes.size() + threadsPerBlock - 1) / threadsPerBlock;
  toLBCode_from_LBGridCoor_Kernel<<<numBlocks, threadsPerBlock, 0, 0>>>(d_innerCodes,
                                                                          d_inner,
                                                                          inner.size(), level);
  cudaDeviceSynchronize();
  cudaMemcpy(innerCodes.data(), d_innerCodes, sizeof(uint64) * inner.size(), cudaMemcpyDeviceToHost);

  cudaFree(d_innerCodes);
  cudaFree(d_inner);


  if (ismultiscale) {
			toMultiscaleCodes_fromSinglescaleCodes(h_borderCodes);
			toMultiscaleCodes_fromSinglescaleCodes(innerCodes);
		}

  cudaFree(d_borderCodes);

  return h_borderCodes;
}


static void ScanLineSeedFill_2D(vector<LBGridCoor> boder,
                                vector<LBGridCoor> &inner)
{
  inner.clear();
  if (boder.size() <= 1)
  {
    return;
  }

  int Imin = (int)boder[0].coI;
  int Imax = Imin;
  int Jmin = (int)boder[0].coJ;
  int Jmax = Jmin;
  for (int i = 0; i < boder.size(); i++)
  {
    Imin = min(Imin, int(boder[i].coI));
    Jmin = min(Jmin, int(boder[i].coJ));
    Imax = max(Imax, int(boder[i].coI));
    Jmax = max(Jmax, int(boder[i].coJ));
  }

  int wid_I = Imax - Imin + 4 + 1;
  int wid_J = Jmax - Jmin + 4 + 1;
  vector<LBGridCoor> SceneData;
  SceneData.resize(wid_I * wid_J);
  for (int i = 0; i < wid_I * wid_J; i++)
  {
    int x = i % wid_I;
    int y = i / wid_I;
    SceneData[i].coI = (uint32)(x + Imin - 2);
    SceneData[i].coJ = (uint32)(y + Jmin - 2);
    SceneData[i].level = 0;
    if (x == 0 || x == wid_I - 1 || y == 0 || y == wid_J - 1)
    {
      SceneData[i].level = 1;
    }
  }

  for (int i = 0; i < boder.size(); i++)
  {
    SceneData[((int)boder[i].coJ - Jmin + 2) * wid_I + (int)boder[i].coI -
              Imin + 2]
        .level = 1;
  }

  vector<LBGridCoor> Stk;
  LBGridCoor StartPoint;
  StartPoint.coI = 1 + Imin - 2;
  StartPoint.coJ = 1 + Jmin - 2;
  StartPoint.level = 1;

  Stk.push_back(StartPoint);

  while (!Stk.empty())
  {

    LBGridCoor SeedPoint = Stk[(int)Stk.size() - 1];
    Stk.pop_back();
    SceneData[((int)SeedPoint.coJ - Jmin + 2) * wid_I + (int)SeedPoint.coI -
              Imin + 2]
        .level = 1;

    int count =
        FillLineRight2((int)SeedPoint.coI - Imin + 2,
                       (int)SeedPoint.coJ - Jmin + 2, SceneData, wid_I, wid_J);
    int xRight = (int)SeedPoint.coI - Imin + 2 + count - 1;
    count =
        FillLineleft2((int)SeedPoint.coI - 1 - Imin + 2,
                      (int)SeedPoint.coJ - Jmin + 2, SceneData, wid_I, wid_J);
    int xLeft = (int)SeedPoint.coI - count - Imin + 2;

    SearchLineNewSeed2(Stk, SceneData, xLeft, xRight,
                       (int)SeedPoint.coJ - 1 - Jmin + 2, wid_I, wid_J);
    SearchLineNewSeed2(Stk, SceneData, xLeft, xRight,
                       (int)SeedPoint.coJ + 1 - Jmin + 2, wid_I, wid_J);
  }

  uint32 level_LB = boder[0].level;
  for (int i = 0; i < SceneData.size(); i++)
  {
    if (SceneData[i].level == 0)
    {
      LBGridCoor temp_coor;
      temp_coor.coI = SceneData[i].coI;
      temp_coor.coJ = SceneData[i].coJ;
      temp_coor.level = level_LB;
      inner.push_back(temp_coor);
    }
  }

  return;
}

void getCodesOfPlygon_fixedlevel_detail(vector<LBPoint> coorlist, uint32 level,
                                        vector<uint64> &borderCodes,
                                        vector<uint64> &innerCodes,
                                        bool isBresenham, bool ismultiscale, 
                                        int max_vcodes_num,
                                        int max_allcodes_num)
{
  vector<uint64> vcodes;
  if (coorlist.size() < 3)
  {
    printf("error: Point num < 3 \n");
    borderCodes.clear();
    innerCodes.clear();
    return;
  }

  int num1 = (int)coorlist.size() - 1;
  if (abs(coorlist[0].lat - coorlist[num1].lat) > 1e-7 ||
      abs(coorlist[0].lat - coorlist[num1].lat) > 1e-7)
  {
    coorlist.push_back(coorlist[0]);
  }

  borderCodes = getCodesOfLines_fixedlevel_CUDA(coorlist,
                                                level,
                                                innerCodes,
                                                isBresenham,
                                                ismultiscale,
                                                max_vcodes_num,
                                                max_allcodes_num); // isBresenham=false

  return;
}

vector<uint64> getCodesOfPlygon_fixedlevel(vector<LBPoint> coorlist,
                                           uint32 level, bool ismultiscale, 
                                           int max_vcodes_num,
                                           int max_allcodes_num)
{
  vector<uint64> vcodes;
  vector<uint64> boderCodes, innerCodes;
  getCodesOfPlygon_fixedlevel_detail(coorlist, level, boderCodes, innerCodes,
                                     false, ismultiscale,
                                      max_vcodes_num,
                                      max_allcodes_num);
  vcodes.insert(vcodes.end(), boderCodes.begin(), boderCodes.end());
  vcodes.insert(vcodes.end(), innerCodes.begin(), innerCodes.end());

  if (ismultiscale)
  {
      toMultiscaleCodes_fromSinglescaleCodes(vcodes);
  }
  return vcodes;
}

#endif // ADVANCE_SpaceCode

//} namespace SpaceCode2DSystem
