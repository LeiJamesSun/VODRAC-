%%This is the source code of the VODRAC paper
%%by Lei Sun (leisunjames@126.com)
%%only for academic use

clc;clear all;close all;

%% Parameters to adjust:

noise=0.01; % noise (not to change)

n_ele=1000; % correspondence number (not to change)

outlier_ratio=0.9; % outlier ratio (change from 0-0.99)

[pts_3d,pts_3d_,R_gt,t_gt]=Build_Scenario(n_ele,noise,outlier_ratio,1,0);


%% VODRAC algorithm:

tic;

rep=0;max_rep=1e+5;cc=0;best_size=0;adj_mat=-ones(n_ele,n_ele);
while 1
    rep=rep+1;
    this_pt=randperm(1*n_ele,2);
    
    if adj_mat(this_pt(1),this_pt(2))==-1
        base=pts_3d(this_pt(1),:)-pts_3d(this_pt(2),:);
        base_=pts_3d_(this_pt(1),:)-pts_3d_(this_pt(2),:);
        dis=tbvalue(abs(norm(base)-norm(base_)),5*noise);
        adj_mat(this_pt(1),this_pt(2))=dis;
        adj_mat(this_pt(2),this_pt(1))=dis;
    end
    
    if adj_mat(this_pt(1),this_pt(2))>0
        co=0;pot_set=zeros(1,1);
        for i=1:n_ele
            if i~=this_pt(1) && i~=this_pt(2)
                if adj_mat(i,this_pt(1))==-1
                    dis=tbvalue(abs(norm(pts_3d(i,:)-pts_3d(this_pt(1),:))-norm(pts_3d_(i,:)-pts_3d_(this_pt(1),:))),5*noise);
                    adj_mat(this_pt(1),i)=dis;
                    adj_mat(i,this_pt(1))=dis;
                end
                if adj_mat(i,this_pt(2))==-1
                    dis=tbvalue(abs(norm(pts_3d(i,:)-pts_3d(this_pt(2),:))-norm(pts_3d_(i,:)-pts_3d_(this_pt(2),:))),5*noise);
                    adj_mat(this_pt(2),i)=dis;
                    adj_mat(i,this_pt(2))=dis;
                end
                if adj_mat(i,this_pt(1))>0
                    if adj_mat(i,this_pt(2))>0
                       co=co+1;
                       pot_set(co)=i;
                    end
                end
            end
        end
        if co+2>=max([0.01*n_ele,best_size])
            diff_mat=zeros(co,co);
            for i=1:co-1
                for j=i+1:co
                    if adj_mat(pot_set(i),pot_set(j))==-1
                        dis=tbvalue(abs(norm(pts_3d(pot_set(i),:)-pts_3d(pot_set(j),:))-norm(pts_3d_(pot_set(i),:)-pts_3d_(pot_set(j),:))),5*noise);
                        adj_mat(pot_set(i),pot_set(j))=dis;
                        adj_mat(pot_set(j),pot_set(i))=dis;
                    end
                    diff_mat(i,j)=adj_mat(pot_set(i),pot_set(j));
                    diff_mat(j,i)=diff_mat(i,j);
                end
            end
            sum_res=zeros(1,co);
            for i=1:co
                sum_res(i)=sum(diff_mat(i,:));
            end
            [~,num_res]=sort(sum_res,'descend');
            inn_rep=min([10,round(co/2)+1]);
            store_T=zeros(3,4,inn_rep);pot_set_=[pot_set,this_pt];error=zeros(1,inn_rep);
            best_consensus_size=0;best_consensus=ones(1,1);
            for i=1:inn_rep
                cc=cc+1;
                Aa=[this_pt,pot_set(num_res(i))];
                [R_raw,t_raw]=Horn_minimal_(pts_3d(Aa,:),pts_3d_(Aa,:));
                store_T(:,:,i)=[R_raw,t_raw];
                consensus=ones(1,1);coo=0;
                for j=1:co
                    re_this=norm(R_raw*pts_3d(pot_set_(j),:)'+t_raw-pts_3d_(pot_set_(j),:)');
                    if re_this<=6*noise
                        coo=coo+1;
                        consensus(coo)=pot_set_(j);
                    end
                end
                if coo>=best_consensus_size
                    best_consensus_size=coo+2;
                    best_consensus=[consensus,this_pt];
                end
            end

if best_consensus_size>=5
    
q_=zeros(3,1);
p_=zeros(3,1);
opt_set=best_consensus;

len_o=length(opt_set);

for i=1:len_o

q_(1)=q_(1)+pts_3d_(opt_set(i),1);
q_(2)=q_(2)+pts_3d_(opt_set(i),2);
q_(3)=q_(3)+pts_3d_(opt_set(i),3);

p_(1)=p_(1)+pts_3d(opt_set(i),1);
p_(2)=p_(2)+pts_3d(opt_set(i),2);
p_(3)=p_(3)+pts_3d(opt_set(i),3);

end

p_=p_/len_o;
q_=q_/len_o;

H=zeros(3,3);
for i=1:len_o
    H=H+(pts_3d(opt_set(i),:)'-p_)*(pts_3d_(opt_set(i),:)'-q_)';
end

[U,~,V]=svd(H);

R_opt=V*U';

t_opt=q_-R_opt*p_;

res=zeros(1,n_ele);cou=0;ok_set=zeros(1,1);
    
for i=1:n_ele
        res(i)=(1*R_opt*((pts_3d(i,:)))'+t_opt-(pts_3d_(i,:))')'*(1*R_opt*((pts_3d(i,:)))'+t_opt-(pts_3d_(i,:))');
        if sqrt(res(i))<=6*noise
            cou=cou+1;
            ok_set(cou)=i;
        end
end

%             [~,err_num]=sort(error,'descend');
%             R_best=store_T(:,1:3,err_num(1));
%             t_best=store_T(:,4,err_num(1));
%             res=R_best*pts_3d(pot_set,:)'+t_best-pts_3d_(pot_set,:)';
%             cou=0;ok_set=zeros(1,1);
%                 for j=1:co
%                     res_this=res(:,j)'*res(:,j);
%                     if res_this<=(6*noise)^2
%                         cou=cou+1;
%                         ok_set(cou)=pot_set(j);
%                     end
%                 end
                if cou>best_size
                    best_size=cou;
                    best_set=ok_set;
                    max_rep=log(0.01)/log(1-(best_size/n_ele)^2);
                end
                
end
        end
    end
    if rep>=max_rep
        break
    end
end

opt_set=best_set;

q_=zeros(3,1);
p_=zeros(3,1);

len_o=length(opt_set);

for i=1:len_o

q_(1)=q_(1)+pts_3d_(opt_set(i),1);
q_(2)=q_(2)+pts_3d_(opt_set(i),2);
q_(3)=q_(3)+pts_3d_(opt_set(i),3);

p_(1)=p_(1)+pts_3d(opt_set(i),1);
p_(2)=p_(2)+pts_3d(opt_set(i),2);
p_(3)=p_(3)+pts_3d(opt_set(i),3);

end

p_=p_/len_o;
q_=q_/len_o;

s_best=1;

H=zeros(3,3);
for i=1:len_o
    H=H+(pts_3d(opt_set(i),:)'-p_)*(pts_3d_(opt_set(i),:)'-q_)';
end

[U,~,V]=svd(H);

R_opt=V*U';

t_opt=q_-s_best*R_opt*p_;

time=toc();

R_error=GetAngularError(R_opt,R_gt)*180/pi;

t_error=norm(t_opt-t_gt');

disp('Rotation Error: ')
disp(R_error)
disp('Translation Error: ')
disp(t_error)
disp('Runtime: ')
disp(time)

%% Functions:

function value = tbvalue(res,th)
    if res>th
        value=0;
    elseif res<=th
        value=(1-res^2/(th^2))^2;
    end
end
