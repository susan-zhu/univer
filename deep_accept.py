import os,json

path=r'D:\anconda_workspace\ai_infra_vllm_test\uniVer\results\records_mt2.jsonl'
for method in ["greedy","univer","RRSw","traversal_verification"]:
    print(f"开始类别{method}:")
    res=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            tmp=json.loads(line)
            if tmp["verify_method"]==method:
                res.append(tmp)

    rounds=0
    depth_accept_dict={}
    for d in res:
        rounds+=d["verification_rounds"]
        if sum(d["depth_accept"].values())!=d["verification_rounds"]:
            print("not eaue")
        for key,value in d["depth_accept"].items():
            depth_accept_dict[key]=depth_accept_dict.get(key,0)+value
    # print(res[0]["depth_accept"],res[0]["verification_rounds"])
    print(rounds)
    sum_v=sum(depth_accept_dict.values())
    print(depth_accept_dict)
    print(sum_v)
    new_v=sum_v

    for k,v in depth_accept_dict.items():
        print(k,v,new_v-v,(new_v-v)/new_v)
        new_v=new_v-v


    # for k in range(0,5):
    #     if k==0:
    #         print(depth_accept_dict[str(k)] / rounds)
    #     else:
    #         print(depth_accept_dict[str(k)]/depth_accept_dict[str(k-1)])


# from itertools import accumulate
#
#
# depths=[1570, 1689, 1385, 983, 777, 701]
# total =  list(accumulate(depths[::-1]))
# total=total[::-1]
# print(total)

