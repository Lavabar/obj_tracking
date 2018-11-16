input:
    strength as floating point >= 0.  0 = no change, high numbers equal stronger correction.
    zoom as floating point >= 1.  (1 = no change in zoom)

algorithm:

    set halfWidth = imageWidth / 2
    set halfHeight = imageHeight / 2
    
    if strength = 0 then strength = 0.00001
    set correctionRadius = squareroot(imageWidth ^ 2 + imageHeight ^ 2) / strength

    for each pixel (x,y) in destinationImage
        set newX = x - halfWidth
        set newY = y - halfHeight

        set distance = squareroot(newX ^ 2 + newY ^ 2)
        set r = distance / correctionRadius
        
        if r = 0 then
            set theta = 1
        else
            set theta = arctangent(r) / r

        set sourceX = halfWidth + theta * newX * zoom
        set sourceY = halfHeight + theta * newY * zoom

        set color of pixel (x, y) to color of source image pixel at (sourceX, sourceY)