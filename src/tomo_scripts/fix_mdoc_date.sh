#!/bin/bash
mv tomo tomo-old ;
mkdir tomo;
for tomo in tomo-old/*.mdoc ; do
awk 'BEGIN{date="";k=0} k==0{if($0~/T = /){split($0,t," ");for(i in t){if(t[i]~/..-*-../){split($0,s,t[i]);split(t[i],d,"-");split(d[3],y,"");date=d[1]"-"d[2]"-"y[length(y)-1]""y[length(y)]}};print s[1],date,s[2];k=1;next}} {if($1~/DateTime/){split($0,l,$3); print l[1]""date""l[2];next}} {print $0}' ${tomo} > tomo/$(basename ${tomo}) ;
done ;
