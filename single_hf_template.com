%chk=${title}.chk
#p UHF/6-31G ${freq} scf(xqc)

${title}

${charge} ${multi}
${atoms}



