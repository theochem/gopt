%chk=${title}.chk
#p UHF/6-31++G* ${freq} scf(xqc)

${title}

${charge} ${multi}
${atoms}



