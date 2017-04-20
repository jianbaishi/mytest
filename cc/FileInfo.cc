#include <fstream>
#include <cstdlib>
#include <cstring>
#include "FileInfo.h"

#define APPEND_NAMEFILE(name, end) #name#end
#define MAX_LINE 512
#define NAME_SHH  "Shanghai"
#define NAME_SHZH "Shenzhen"
#define NAME_CHYB "Chuangyeban"
#define NAME_MYBK "MYBK"

using std::string;
using std::wstring;
using std::cout;
using std::wcout;
using std::endl;
using std::pair;
using std::ifstream;
using std::make_pair;

struct NameCodeMap {
    string name;
    string code;
};

NameCodeMap g_name_code_map[] =
{
    {NAME_SHH, "000001"},
    {NAME_SHZH, "399001"},
    {NAME_CHYB, "399006"},
    {NAME_MYBK, "300188"}
};

string *get_code_from_map(const char *name)
{
    int i;
    int size;
    size = sizeof(g_name_code_map) / sizeof(NameCodeMap);
    for (i = 0; i < size; i++) {
        if (strcmp(name, g_name_code_map[i].name.c_str()) == 0) {
            return &g_name_code_map[i].code;
        }
    }

    return NULL;
}

string *get_name_from_map(const char *code)
{
    int i;
    int size;
    size = sizeof(g_name_code_map) / sizeof(NameCodeMap);
    for (i = 0; i < size; i++) {
        if (strcmp(code, g_name_code_map[i].code.c_str()) == 0) {
            return &g_name_code_map[i].name;
        }
    }

    return NULL;
}

namespace fileinfo {
void ShareBaseData::operator=(const ShareBaseData &data)
{
    code = data.code;
    name = data.name;
    date = data.date;
    place = data.place;
    old_end_value = data.old_end_value;
    new_start_value = data.new_start_value;
    total_number = data.total_number;
    high_value = data.high_value;
    low_value = data.low_value;
    new_end_value = data.new_end_value;
    total_pen = data.total_pen;
    change = data.change;
    gmv = data.gmv;
}

int ShareBaseData::GetShareBaseData(const char *filename, int line)
{
    string path = "data/";
    string file = path + filename;
    ifstream f_in(file);
    string str;
    int count;
    char *p, *q;
    char *tmpstr;
    int pos;

    count = 0;
    while(getline(f_in,str)) {
        count++;
        if (line == count) {
            break;
        }
    }
    cout << line << " " << count << endl;
    if (line != count) {
        return -2;
    }

    /* ,, 处理 */
    pos = 0;
    while (1) {
        pos = str.find(",,", pos + 1);
        if(pos == string::npos) {
            break;
        }
        str.insert(pos + 1, "0");
    }


    tmpstr = new char[str.size() + 1]();
    strncpy(tmpstr, str.c_str(), str.size());
    cout << tmpstr << endl;

    /* "分割处理 */
    p = strtok(tmpstr, "\"");
    if (p == NULL) {
        return -1;
    }
    code = p;

    p = strtok(NULL, "\"");
    if (p == NULL) {
        return -1;
    }

    /* ,分割处理 */
    q = strtok(p, ",");
    if (q == NULL) {
        return -1;
    }
    name = q;

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    date = q;

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }

    place = q;

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    old_end_value = atof(q);

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    new_start_value = atof(q);

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    total_number = atoi(q);

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    high_value = atof(q);

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    low_value = atof(q);

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    new_end_value = atof(q);

    q = strtok(NULL, ",");
    if (q == NULL) {
        total_pen = 0;
    } else {
        total_pen = atoi(q);
    }

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    change = atof(q);

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    gmv = atof(q);
    return 0;
}

int ShareBaseData::GetShareBaseData(ShareBaseData &output, const char *filename, int line)
{
    ShareBaseData data;
    string path = "data/";
    string file = path + filename;
    ifstream f_in(file);
    string str;
    int count;
    char *p, *q;
    char *tmpstr;
    int pos;

    count = 0;
    while(getline(f_in,str)) {
        count++;
        if (line == count) {
            break;
        }
    }

    if (line != count) {
        return -2;
    }

    /* ,, 处理 */
    pos = 0;
    while (1) {
        pos = str.find(",,", pos + 1);
        if(pos == string::npos) {
            break;
        }
        str.insert(pos + 1, "0");
    }


    tmpstr = new char[str.size() + 1]();
    strncpy(tmpstr, str.c_str(), str.size());
    cout << tmpstr << endl;

    /* "分割处理 */
    p = strtok(tmpstr, "\"");
    if (p == NULL) {
        return -1;
    }
    data.code = p;

    p = strtok(NULL, "\"");
    if (p == NULL) {
        return -1;
    }

    /* ,分割处理 */
    q = strtok(p, ",");
    if (q == NULL) {
        return -1;
    }
    data.name = q;

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    data.date = q;

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }

    data.place = q;

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    data.old_end_value = atof(q);

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    data.new_start_value = atof(q);

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    data.total_number = atoi(q);

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    data.high_value = atof(q);

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    data.low_value = atof(q);

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    data.new_end_value = atof(q);

    q = strtok(NULL, ",");
    if (q == NULL) {
        data.total_pen = 0;
    } else {
        data.total_pen = atoi(q);
    }

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    data.change = atof(q);

    q = strtok(NULL, ",");
    if (q == NULL) {
        return -1;
    }
    data.gmv = atof(q);

    output = data;
    return 0;
}

void ShareBaseData::PrintShareBaseData() const
{
    char p[] = ",";
    cout << code << p;
    cout << name ;
    cout << p << date << p ;
    cout << place;
    cout << p;
    cout << old_end_value << p << new_start_value << p << total_number << p;
    cout << high_value << p << low_value << p << new_end_value << p;
    cout << total_pen << p << change << p << gmv << endl;
}

ShareData::ShareData(const char *name1)
{
    ShareBaseData *data1 = new ShareBaseData;
    Add(name1, data1);
}

ShareData::ShareData(const char *name1, const char *name2)
{
    ShareBaseData *data1 = new ShareBaseData;
    Add(name1, data1);
    ShareBaseData *data2 = new ShareBaseData;
    Add(name2, data2);
}

ShareData::ShareData(const char *name1, const char *name2, const char *name3)
{
    ShareBaseData *data1 = new ShareBaseData;
    Add(name1, data1);
    ShareBaseData *data2 = new ShareBaseData;
    Add(name2, data2);
    ShareBaseData *data3 = new ShareBaseData;
    Add(name3, data3);
}

ShareData::ShareData(const char *name1, const char *name2, const char *name3, const char *name4)
{
    ShareBaseData *data1 = new ShareBaseData;
    Add(name1, data1);
    ShareBaseData *data2 = new ShareBaseData;
    Add(name2, data2);
    ShareBaseData *data3 = new ShareBaseData;
    Add(name3, data3);
    ShareBaseData *data4 = new ShareBaseData;
    Add(name4, data4);
}

ShareData::~ShareData()
{
    map<string, ShareBaseData>::const_iterator map_it;
    map_it = share_value.begin();
    while (map_it != share_value.end()) {
        share_value.erase(map_it);
        //delete &map_it->second;
        map_it = share_value.begin();
    }
}

bool ShareData::Check()
{
    ShareBaseData data;
    string b_date;
    map<string, ShareBaseData>::const_iterator map_it = share_value.begin();
    if (map_it != share_value.end()) {
        b_date = map_it->second.date;
        ++map_it;
    } else {
        return false;
    }

    if (map_it == share_value.end()) {
        return false;
    }

    while (map_it != share_value.end()) {
        data = map_it->second;
        if (data.date != b_date) {
            return false;
        }
        ++map_it;
    }

    return true;
}

int ShareData::SetData(const char *sharename, const char *filename, int line)
{
    int ret;
    map<string, ShareBaseData>::iterator map_it = share_value.find(sharename);
    if (map_it == share_value.end()) {
        return -1;
    }

    cout << "GetShareBaseData: " << endl;
    ret = map_it->second.GetShareBaseData(filename, line);
    if (ret >= 0) {
        map_it->second.PrintShareBaseData();
    }

    return ret;
}

int ShareData::Add(const char *name, ShareBaseData *data)
{
    pair<map<string, ShareBaseData>::iterator, bool> ret = share_value.insert(make_pair(name, *data));
    if (!ret.second) {
        return -2;
    }

    return 0;
}

int ShareData::Delete(string &name)
{
    map<string, ShareBaseData>::iterator map_it = share_value.find(name);
    if (map_it == share_value.end()) {
        return -1;
    }
    share_value.erase(map_it);
    return 0;
}

void ShareData::PrintShareData() const
{
    map<string, ShareBaseData>::const_iterator map_it = share_value.begin();
    while (map_it != share_value.end()) {
        cout << "    " << map_it->first << " : ";
        map_it->second.PrintShareBaseData();
        map_it++;
    }
}

ShareAllData::~ShareAllData()
{
    vector<ShareData>::iterator iter = alldata.begin();
    while (iter != alldata.end()) {
        alldata.erase(iter);
        iter++;
    }
}

int ShareAllData::AddOneData(const string filename_end, int line)
{
    int ret;
    ShareData *sdata;
    string tmp_head = "sh_hq_";
    string tmp_file_str, *tmp_code_str;
    const char *tmp_str;

    switch (args_num) {
    case 4:
        sdata = new ShareData(name1.c_str(), name2.c_str(), name3.c_str(), name4.c_str());
        break;
    case 3:
        sdata = new ShareData(name1.c_str(), name2.c_str(), name3.c_str());
        break;
    case 2:
        sdata = new ShareData(name1.c_str(), name2.c_str());
        break;
    case 1:
        sdata = new ShareData(name1.c_str());
        break;
    default:
        cout << "Error args_num: " << args_num << endl;
        return -1;
    }

    switch (args_num) {
    case 4:
        tmp_str = name4.c_str();
        tmp_code_str = get_code_from_map(tmp_str);
        if (strncmp("3", tmp_code_str->c_str(), 1) == 0) {
            tmp_head = "sz_hq_";
        } else {
            tmp_head = "sh_hq_";
        }
        tmp_file_str = tmp_head + *tmp_code_str + filename_end;
        cout << tmp_str << " " << *tmp_code_str << ", filename: " << tmp_file_str <<  endl;
        ret = sdata->SetData(tmp_str, tmp_file_str.c_str(), line);
        if (ret < 0) {
            cout << "Failed to set " << tmp_str << " in line " << line << endl;
            return ret;
        }
    case 3:
        tmp_str = name3.c_str();
        tmp_code_str = get_code_from_map(tmp_str);
        if (strncmp("3", tmp_code_str->c_str(), 1) == 0) {
            tmp_head = "sz_hq_";
        } else {
            tmp_head = "sh_hq_";
        }
        tmp_file_str = tmp_head + *tmp_code_str + filename_end;
        cout << tmp_str << " " << *tmp_code_str << ", filename: " << tmp_file_str <<  endl;
        ret = sdata->SetData(tmp_str, tmp_file_str.c_str(), line);
        if (ret < 0) {
            cout << "Failed to set " << tmp_str << " in line " << line << endl;
            return ret;
        }
    case 2:
        tmp_str = name2.c_str();
        tmp_code_str = get_code_from_map(tmp_str);
        if (strncmp("3", tmp_code_str->c_str(), 1) == 0) {
            tmp_head = "sz_hq_";
        } else {
            tmp_head = "sh_hq_";
        }
        tmp_file_str = tmp_head + *tmp_code_str + filename_end;
        cout << tmp_str << " " << *tmp_code_str << ", filename: " << tmp_file_str <<  endl;
        ret = sdata->SetData(tmp_str, tmp_file_str.c_str(), line);
        if (ret < 0) {
            cout << "Failed to set " << tmp_str << " in line " << line << endl;
            return ret;
        }
    case 1:
        tmp_str = name1.c_str();
        tmp_code_str = get_code_from_map(tmp_str);
        if (strncmp("3", tmp_code_str->c_str(), 1) == 0) {
            tmp_head = "sz_hq_";
        } else {
            tmp_head = "sh_hq_";
        }
        tmp_file_str = tmp_head + *tmp_code_str + filename_end;
        cout << tmp_str << " " << *tmp_code_str << ", filename: " << tmp_file_str <<  endl;
        ret = sdata->SetData(tmp_str, tmp_file_str.c_str(), line);
        if (ret < 0) {
            cout << "Failed to set " << tmp_str << " in line " << line << endl;
            return ret;
        }
        break;
    default:
        cout << "Error args_num: " << args_num << endl;
        return -1;
    }

    ret = sdata->Check();
    if (ret < 0) {
        cout << "Failed check" << endl;
        return ret;
    }

    /* 插到alldata最后 */
    alldata.push_back(*sdata);

    return 0;
}

int ShareAllData::AddAllData(const string filename_end)
{
    int line;
    int ret;

    for (line = 1; line < MAX_LINE; line++) {
        ret = AddOneData(filename_end, line);
        if (ret == -2) {
            return 0;
        }
    }

    return ret;
}

void ShareAllData::PrintShareAllData() const
{
    int count;
    vector<ShareData>::const_iterator iter = alldata.begin();
    count = 0;
    while (iter != alldata.end()) {
        cout << ++count << ": " << endl;
        (*iter).PrintShareData();
        iter++;
    }
}

}

int main()
{
    int ret;
    fileinfo::ShareAllData adata(NAME_MYBK, NAME_CHYB, NAME_SHZH, NAME_SHH);

    adata.AddAllData("_2011.csv");
    adata.AddAllData("_2012.csv");
    adata.AddAllData("_2013.csv");
    adata.AddAllData("_2014.csv");
    adata.AddAllData("_2015.csv");
    adata.AddAllData("_2016.csv");
    adata.AddAllData("_2017.csv");
    adata.PrintShareAllData();

    return 0;
}
