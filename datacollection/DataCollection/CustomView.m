//
//  CustomView.m
//  DataCollection
//
//  Created by Frederik Jensen on 30/10/16.
//  Copyright (c) 2016 Brinck10. All rights reserved.
//

#import "CustomView.h"

@implementation CustomView {
}
@synthesize delegate;

- (id)initWithFrame:(NSRect)frameRect {
    self = [super initWithFrame:frameRect];
    if (self) {
        
    }
    return self;
}

- (void)drawRect:(NSRect)dirtyRect {
    [super drawRect:dirtyRect];
    
    CGContextRef ctx = [[NSGraphicsContext currentContext] graphicsPort];
    CGContextClearRect(ctx, NSMakeRect(0, 0, 40, 40));
    CGContextSetFillColorWithColor(ctx, [[NSColor blueColor] CGColor]);
    CGContextFillRect(ctx, NSMakeRect(0, 0, 40,40));
}



@end