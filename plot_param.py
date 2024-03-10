import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
import csv
import math
from datetime import datetime

val = [[0.4301246713843673], [0.4691853857704785], [0.4008283331700422], [0.4024187207808016], [0.48497747596123136], [0.5012559012298312], [0.4507093793109166], [0.49079028204188946], [0.5218222728640749], [0.5747042729472002], [0.5211582321513829], [0.5997767078471755], [0.6114171687545671], [0.5712133952685834], [0.5787636582917378], [0.5233488494449486], [0.5406313836768498], [0.4970403633769273], [0.472851651114803], [0.5262840418480107], [0.552580249598362], [0.5419804877700475], [0.5209668849709784], [0.6049467965131025], [0.5605255508697577], [0.5872879179691577], [0.6939490273022528], [0.4715759545574824], [0.5813861257366841], [0.5710982120592745], [0.5806122460041905], [0.5092919381896369], [0.5624705213834427], [0.6037694546856882], [0.5097743787945951], [0.5828812764864755], [0.608973879982912], [0.5551935000530965], [0.4940005743610407], [0.5125811094375822], [0.5069690713891756], [0.48221335548545907], [0.4958305385863355], [0.5103450160299535], [0.5898914901666664], [0.5832124615005251], [0.5435969228677229], [0.4950348587570781], [0.4825335804863511], [0.5332777242011422], [0.5070362457256531], [0.5900093968999057], [0.5757447387768807], [0.5037735964871071], [0.5745041052676683], [0.5538154269497148], [0.46655804045751864], [0.5579592275433365], [0.5337809852366877], [0.5715822056659388], [0.48863597203687625], [0.466844451538851], [0.541166453581506], [0.5243102057692384], [0.476232546020678], [0.4686994968925319], [0.5364407876951135], [0.6006674002889922], [0.5064628883839566], [0.6092625858343333], [0.5440131242185928], [0.5316374328531194], [0.5651729339640585], [0.6128690890474928], [0.5340639080315308], [0.5359050336682641], [0.6228646027739354], [0.554958998015067], [0.5990560837242943], [0.6420271648080614], [0.5734665242654038], [0.4984659415393477], [0.5918904026925184], [0.5454369919268044], [0.4764662075542806], [0.5910642307557151], [0.595453233850451], [0.5452309709113153], [0.5617734498521554], [0.5747991187475382], [0.5506473427278684], [0.5562486863777922], [0.6102874332142934], [0.49675575430355606], [0.5176319535944198], [0.4600494802079781], [0.5691867702006862], [0.552905296400737], [0.5558828561378363], [0.5909992509658043]]
val_accpt = [[0.33801247771836], [0.40363636363636374], [0.4334581105169341], [0.3879411764705884], [0.37917112299465244], [0.3670855614973262], [0.3197593582887701], [0.3534491978609627], [0.3224509803921568], [0.3371390374331552], [0.36341354723707653], [0.41029411764705886], [0.31538324420677355], [0.34377896613190734], [0.4622816399286988], [0.3182798573975044], [0.41244206773618536], [0.3814081996434938], [0.43147950089126563], [0.3761319073083778], [0.4436274509803921], [0.4202584670231728], [0.4299643493761141], [0.3566488413547238], [0.4037611408199645], [0.4264705882352942], [0.37213903743315513], [0.43934937611408204], [0.4320320855614973], [0.44719251336898386], [0.40320855614973267], [0.3605258467023174], [0.3665151515151515], [0.4062299465240642], [0.2825846702317291], [0.3897237076648841], [0.40434046345811053], [0.409554367201426], [0.3651247771836008], [0.42345811051693416], [0.4745811051693405], [0.48538324420677353], [0.45191622103386814], [0.4182442067736186], [0.3899554367201426], [0.3544206773618538], [0.3617736185383244], [0.291096256684492], [0.4884848484848485], [0.4341265597147951]]
urllc_accpt_ddpg = [[0.36752136752136755], [0.4731182795698925], [0.36936936936936937], [0.45161290322580644], [0.4659090909090909], [0.44329896907216493], [0.44565217391304346], [0.43956043956043955], [0.4], [0.4056603773584906]]
urllc_accpt_reinf = [[0.5663716814159292], [0.417910447761194], [0.7804878048780488], [0.8076923076923077], [0.5625], [0.6486486486486487], [0.5656565656565656], [0.7227722772277227], [0.5588235294117647], [0.8369565217391305]]


def plot_param_one_rep(param, repitition, episode, name=""):
    # print(param)
    my_list = []
    for ep in range(episode):
        my_list.append(param[ep][0])
        # req_param = param[ep]
        # plt.plot(req_param)
        # for rep in range(repitition):
        #     print(req_param[rep])
    # print(my_list)
    average = sum(my_list)/len(my_list)
    now = datetime.now()
    current_time = now.strftime("%d-%m-%Y:%H:%M:%S")
    plt.plot(my_list)
    plt.savefig("./plots/plots_"+ name + "_" + current_time +".png")
    # plt.show()
    return average

def plot_param_multi_rep(param, repitition, episode, name=""):
    now = datetime.now()
    current_time = now.strftime("%d-%m-%Y-%H:%M:%S")
    # print("Current Time =", current_time)
    ep_avg_list = []
    for ep in range(episode):
        ep_param = param[ep]
        ep_avg_list.append(sum(ep_param)/len(ep_param))
        plt.plot(ep_param)
    # plt.savefig("./plots/plots_"+ name + "_" + current_time +".png")
    # print("Average:",sum(ep_avg_list)/len(ep_avg_list))
    avg = sum(ep_avg_list)/len(ep_avg_list)
    return str(round(avg,4)) + "\n"
    # plt.show()

# plot_param_multi_rep(val,20,1)
plot_param_one_rep(val,1,100)
# print(sum(val)/len(val)) 