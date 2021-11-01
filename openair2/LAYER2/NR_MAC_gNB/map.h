#ifndef _MAP_H_
#define _MAP_H_
typedef struct {
    uint8_t slice_id;
    uint8_t slice_online;
    uint8_t slice_name[16*6];
    uint8_t rbstartlocation;
    uint8_t rboverlocation;
    uint8_t ueid[5];    
}slicing;


extern slicing slices[3];
extern int vrb_map_new[3][20][106];
extern int count;
extern int rbstart_new;
extern int rbrbstart_new_count;
extern uint8_t ue_speed[25];
#endif





