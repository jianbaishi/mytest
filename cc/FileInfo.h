#ifndef __FILEINFO_H__
#define __FILEINFO_H__

#include <functional>
#include <string>
#include <vector>
#include <map>
#include <iostream>

using std::string;
using std::wstring;
using std::map;
using std::vector;

namespace fileinfo {

class ShareBaseData {
public:
    ShareBaseData() {}
    ShareBaseData(string code_, string name_, string date_, string place_, double old_end_value_,
                  double new_start_value_, int total_number_, double high_value_, double low_value_,
                  double new_end_value_, int total_pen_, int change_, double gmv_)
        : code(code_), name(name_), date(date_), place(place_), old_end_value(old_end_value_),
          new_start_value(new_start_value_), total_number(total_number_), high_value(high_value_), low_value(low_value_),
          new_end_value(new_end_value_), total_pen(total_pen_), change(change_), gmv(gmv_){}
    ~ShareBaseData() {}
    void operator=(const ShareBaseData &data);
    int GetShareBaseData(ShareBaseData &output, const char *filename, int line);
    int GetShareBaseData(const char *filename, int line);
    void PrintShareBaseData() const;
    string code;             /* 证券代码 */
    string name;             /* 证券简称 */
    string date;             /* 交易日期 */
    string place;            /* 交易所 */
    double old_end_value;    /* 昨日收盘 */
    double new_start_value;  /* 今日开盘 */
    int total_number;        /* 成交数量 */
    double high_value;       /* 最高成交 */
    double low_value;        /* 最低成交 */
    double new_end_value;    /* 最近成交 */
    int total_pen;           /* 总笔数 */
    double change;              /* 涨跌幅 */
    double gmv;              /* 成交金额 */
};

class ShareData {
public:
    ShareData(const char *name1);
    ShareData(const char *name1, const char *name2);
    ShareData(const char *name1, const char *name2, const char *name3);
    ShareData(const char *name1, const char *name2, const char *name3, const char *name4);
    ~ShareData();
    bool Check();
    int SetData(const char *sharename, const char *filename, int line);
    int Add(const char *name, ShareBaseData *data);
    int Delete(string &name);
    void PrintShareData() const;
    map<string, ShareBaseData> share_value;
};

class ShareAllData {
public:
    ShareAllData() {}
    ShareAllData(const char * in_name1)
        : args_num(1), name1(in_name1) {}
    ShareAllData(const char * in_name1, const char * in_name2)
        : args_num(2), name1(in_name1), name2(in_name2) {}
    ShareAllData(const char * in_name1, const char * in_name2, const char * in_name3)
        : args_num(3), name1(in_name1), name2(in_name2), name3(in_name3) {}
    ShareAllData(const char * in_name1, const char * in_name2, const char * in_name3, const char * in_name4)
        : args_num(4), name1(in_name1), name2(in_name2), name3(in_name3), name4(in_name4) {}
    ~ShareAllData();
    int AddOneData(const string filename_end, int line);
    int AddAllData(const string filename_end);
    void PrintShareAllData() const;

    vector<ShareData> alldata;
    int args_num;
    string name1;
    string name2;
    string name3;
    string name4;
};

}

#endif /* endif __FILEINFO_H__ */
