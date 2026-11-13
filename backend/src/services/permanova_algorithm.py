import math
import numpy as np

def _validate_matrix(clr,labels):
    try:Z=np.asarray(clr,dtype=float)
    except (TypeError,ValueError,OverflowError):raise ValueError("CLR须为有限二维矩阵")
    if Z.ndim!=2 or len(Z)<4 or not 2<=Z.shape[1]<=8 or not np.isfinite(Z).all():raise ValueError("至少4行、2–8维有限CLR")
    if len(labels)!=len(Z) or any(not isinstance(g,str) or not g for g in labels):raise ValueError("每行须有非空字符串组标签")
    labels=np.asarray(labels);groups=sorted(set(labels.tolist()));n=len(Z);k=len(groups)
    if not 2<=k<=5:raise ValueError("要求2–5个预定义组")
    indices={g:np.flatnonzero(labels==g) for g in groups}
    if any(len(i)<2 for i in indices.values()):raise ValueError("每组至少2个独立样本")
    return Z,labels,groups,n,k,indices

def _scale_context(Z):
    n=len(Z);D=Z-Z[0];center=np.array([math.fsum(D[:,j])/n for j in range(Z.shape[1])]);U=D-center;scale=float(np.max(np.abs(U)))
    return center,U,scale

def _group_residuals(Z,indices,center,scale):
    """组内CLR质心残差与原始单位距离；分组锚定避免大数相减损失精度。"""
    V=(Z-Z[0]-center)/scale if scale>0 else Z-Z[0]
    out={}
    for group,ix in indices.items():
        G=V[ix];shift=G-G[0];offset=np.array([math.fsum(shift[:,j])/len(ix) for j in range(Z.shape[1])]);residual=shift-offset
        raw=Z[ix];delta=raw-raw[0];mean=raw[0]+np.array([math.fsum(delta[:,j])/len(ix) for j in range(Z.shape[1])])
        distances=np.sqrt(np.sum(residual**2,axis=1))*scale if scale>0 else np.sqrt(np.sum(residual**2,axis=1))
        out[group]=(residual,mean,distances)
    return out

def permanova_statistics(clr,labels):
    Z,labels,groups,n,k,indices=_validate_matrix(clr,labels)
    center,U,scale=_scale_context(Z)
    if scale==0:raise ValueError("总CLR平方和SST=0，PERMANOVA未定义")
    V=U/scale;sst=math.fsum(float(v*v) for v in V.flat);within_terms=[];summaries=[]
    residuals=_group_residuals(Z,indices,center,scale)
    for group,ix in indices.items():
        residual,mean,distances=residuals[group]
        within=math.fsum(float(v*v) for v in residual.flat);within_terms.append(within)
        exp=np.exp(mean-np.max(mean));composition_center=exp/exp.sum()
        summaries.append({"group":group,"n":len(ix),"mean_clr":mean.tolist(),"aitchison_center":composition_center.tolist(),"mean_distance_to_centroid":float(distances.mean()),"max_distance_to_centroid":float(distances.max())})
    ssw=math.fsum(within_terms)
    ssb=sst-ssw
    if ssb<0 and abs(ssb)<=64*np.finfo(float).eps*sst:ssb=0.
    if ssb<0:raise ValueError("平方和分解数值失败")
    R2=min(1.,max(0.,ssb/sst));F=None;status='unbounded' if ssw==0 else 'finite'
    if ssw>0:
        F=(ssb/(k-1))/(ssw/(n-k))
        if not np.isfinite(F):F=None;status='numerical_overflow'
    return {"n":n,"k":k,"df_between":k-1,"df_within":n-k,"SST":sst*scale*scale,"SSW":ssw*scale*scale,"SSB":ssb*scale*scale,"R2":R2,"F":F,"F_status":status,"groups":summaries,"overall_mean_clr":(Z[0]+center).tolist(),"scale":scale,"method_note":"CLR欧氏平方和；锚定中心、相对缩放及math.fsum组SSW汇总；不用绝对SST阈值。"}

def dispersion_statistics(clr,labels):
    """组内离散度：样本到各自组CLR质心距离的一方ANOVA型R²/pseudo-F。"""
    Z,labels,groups,n,k,indices=_validate_matrix(clr,labels);center,U,scale=_scale_context(Z)
    residual_map=_group_residuals(Z,indices,center,scale) if scale>0 else {group:(np.zeros((len(ix),Z.shape[1])),None,np.zeros(len(ix))) for group,ix in indices.items()}
    distances=np.zeros(n);group_means={};summaries=[]
    for group,ix in indices.items():
        d=np.asarray(residual_map[group][2],dtype=float);distances[ix]=d;mean=math.fsum(float(v) for v in d)/len(ix);group_means[group]=mean
        summaries.append({"group":group,"n":len(ix),"mean_distance_to_centroid":mean,"mean_squared_distance_to_centroid":math.fsum(float(v*v) for v in d)/len(ix),"max_distance_to_centroid":float(np.max(d))})
    zero_tolerance=64*np.finfo(float).eps*max(1.,float(np.max(np.abs(Z))),float(np.max(distances)))*math.sqrt(Z.shape[1])
    all_zero=scale==0 or float(np.max(distances))<=zero_tolerance
    if all_zero:
        for item in summaries:item["mean_distance_to_centroid"]=0.;item["mean_squared_distance_to_centroid"]=0.;item["max_distance_to_centroid"]=0.
        distances[:]=0.;group_means={g:0. for g in groups}
    total_mean=math.fsum(float(v) for v in distances)/n
    ssw=math.fsum(math.fsum(float(v-group_means[group])**2 for v in distances[indices[group]]) for group in groups)
    sst=math.fsum(float(v-total_mean)**2 for v in distances);ssb=sst-ssw
    if ssb<0 and abs(ssb)<=64*np.finfo(float).eps*sst:ssb=0.
    result={"n":n,"k":k,"df_between":k-1,"df_within":n-k,"SST":sst,"SSW":ssw,"SSB":max(0.,ssb),"R2":None,"F":None,"F_status":"not_testable","groups":summaries,"mean_distance":total_mean,"distance_definition":"样本到各自组CLR质心的CLR欧氏距离","status":"testable","reason":"","note":"","method":"PERMDISP-distance-to-centroid","test":"组内离散度置换检验","permutations":None,"seed":None,"null_R2":[],"null_count":0,"invalid_permutations":0,"p":None,"extreme":None,"resolution":None,"comparison_threshold":None,"tie_tolerance":None,"tail":"离散度置换未定义；缺值不作为显著结论。"}
    if all_zero:
        result.update(status="not_testable",reason="所有样本到各自组CLR质心的距离均为0（组内无离散，或组内恒定且完全分离）；离散度差异未定义，不把缺值当显著。")
        return result
    if sst==0:
        result.update(status="not_testable",reason="各组样本到组CLR质心的距离都相同（距离总平方和SST_D=0）；无法比较组内离散度，不把缺值当显著。")
        return result
    if ssb<0:
        result.update(status="not_testable",reason="离散度平方和分解数值失败；不把缺值当显著。")
        return result
    result["R2"]=min(1.,max(0.,ssb/sst))
    if ssw==0:
        result["F_status"]="unbounded";result["note"]="组内距离围绕各自组均值无变化，离散度pseudo-F无界；置换仍比较有限的离散度R²。"
    else:
        F=(ssb/(k-1))/(ssw/(n-k))
        if np.isfinite(F):result["F"]=float(F);result["F_status"]="finite"
        else:result["F_status"]="numerical_overflow";result["note"]="离散度pseudo-F超出数值范围，保留R²置换结论。"
    return result

def upper_tail_p(observed,null,label="R²"):
    values=np.asarray(null,dtype=float)
    if not np.isfinite(observed) or not 0<=observed<=1 or values.ndim!=1 or len(values)==0 or not np.isfinite(values).all() or np.any((values<0)|(values>1)):raise ValueError(f"{label}须为[0,1]有限值且null非空")
    tolerance=64*np.finfo(float).eps*max(1.,abs(float(observed)),float(np.max(np.abs(values))))
    threshold=float(observed-tolerance);extreme=int(np.sum(values>=threshold))
    return {"p":(extreme+1)/(len(values)+1),"extreme":extreme,"resolution":1/(len(values)+1),"comparison_threshold":threshold,"tie_tolerance":float(tolerance),"tail":f"{label}上尾含并列，+1；并列保护为64×机器epsilon×max(1,观测与null绝对{label})，不用于SST拒绝。"}

def _finish_dispersion(observed,null,permutations,seed,invalid=0):
    block=dict(observed)
    block.update({"permutations":permutations,"seed":seed,"null_R2":list(null),"null_count":len(null),"invalid_permutations":int(invalid),"shared_basis":"与组间PERMANOVA共用冻结样本、显式零替换与同一CLR矩阵；同一种子下共用标签置换序列，每次置换重算各组CLR质心和样本距离。","assumptions":"组标签在零假设下可交换，置换保持各组样本数（含不等组规模）；每次置换重算组质心与距离。独立样本是前提；本检验不证明工艺因果，也不把不显著当作离散度同质。","interpretation":"统计量为距离值的一方ANOVA型R²/pseudo-F，上尾置换含并列。p小表示各组离散度可能不同，须与组间差异分别解读；p不小不证明离散度同质。"})
    if observed["status"]!="testable" or not null:
        reason=observed["reason"] or "没有可用的离散度置换结果，无法计算置换p；不把缺值当显著。"
        block.update({"status":"not_testable","p":None,"extreme":None,"resolution":1/(permutations+1),"comparison_threshold":None,"tie_tolerance":None,"tail":"离散度置换未定义；缺值不作为显著结论。","reason":reason})
        if observed["status"]=="testable":block.update({"reason":f"观测离散度可计算，但{invalid or permutations}次置换均未得到有限离散度R²；无法完成置换检验，不把缺值当显著。"})
        return block
    tail=upper_tail_p(observed["R2"],null,label="离散度R²")
    reason=f"另有{invalid}次置换距离分解无定义，已从分母排除且不计为极端；不把缺值当显著。" if invalid else ""
    block.update({"status":"testable","p":tail["p"],"extreme":tail["extreme"],"resolution":tail["resolution"],"comparison_threshold":tail["comparison_threshold"],"tie_tolerance":tail["tie_tolerance"],"tail":tail["tail"],"reason":reason})
    return block

def _validate_permutation_args(permutations,seed):
    if type(permutations) is not int or not 199<=permutations<=999:raise ValueError("置换数须为199–999整数")
    if type(seed) is not int or not 0<=seed<=2147483647:raise ValueError("种子须为非负32位整数")

def dispersion_permutation_test(clr,labels,permutations=199,seed=2026):
    _validate_permutation_args(permutations,seed)
    if len(labels)*permutations>150000:raise ValueError("样本数×置换数不能超过150000")
    observed=dispersion_statistics(clr,labels);rng=np.random.default_rng(seed);null=[];invalid=0
    if observed["status"]=="testable":
        for _ in range(permutations):
            item=dispersion_statistics(clr,rng.permutation(labels).tolist())
            if item["status"]=="testable":null.append(item["R2"])
            else:invalid+=1
    return _finish_dispersion(observed,null,permutations,seed,invalid)

def permutation_test(clr,labels,permutations=199,seed=2026,include_dispersion=False):
    _validate_permutation_args(permutations,seed)
    if len(labels)*permutations>150000:raise ValueError("样本数×置换数不能超过150000")
    observed=permanova_statistics(clr,labels)
    dispersion_observed=dispersion_statistics(clr,labels) if include_dispersion else None
    rng=np.random.default_rng(seed);null=[];null_dispersion=[];invalid_dispersion=0
    for _ in range(permutations):
        permuted=rng.permutation(labels).tolist()
        null.append(permanova_statistics(clr,permuted)['R2'])
        if include_dispersion and dispersion_observed["status"]=="testable":
            item=dispersion_statistics(clr,permuted)
            if item["status"]=="testable":null_dispersion.append(item["R2"])
            else:invalid_dispersion+=1
    out={**observed,"null_R2":null,"permutations":permutations,**upper_tail_p(observed['R2'],null)}
    if include_dispersion:out["dispersion"]=_finish_dispersion(dispersion_observed,null_dispersion,permutations,seed,invalid_dispersion)
    return out
