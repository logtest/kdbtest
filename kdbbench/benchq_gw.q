\p 22223
\c 33 333
gw: hopen `::12288;

gendata:{[n;nsym]
 times:([] time: 2025.01.01 + 0D00:05:00 *til `int$n%nsym);
 symbols:nsym# distinct ([] sym:n?`3);
 d:`sym`time xasc symbols cross times;
 d: update open:100+sums -0.5+n?1f, volume:n?1000 from d;
 d: update high:open+n?1f, low:open-n?1f from d;
 d: update close:0.5*high+low, turnover:open*volume from d;
 :d;
 };


data1:gendata[10000;100];
gw "0!`d"; //make sure d is not keyed

timefetchdata: value "\\t d:gw \"d\"";
num:count d;
timeupsert: value"\\t do[1000;gw({`d upsert x};data1)]"
gw "`sym`time xkey `d";
timeupsertkey: value"\\t do[10;gw({`d upsert x};data1)]"

result: gw "result";
result:result,'enlist `numgw`timefetchdatagw`timeupsertgw`timeupsertkeygw!(num,timefetchdata,timeupsert,timeupsertkey);
show result


